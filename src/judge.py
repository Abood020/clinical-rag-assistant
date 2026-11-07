"""
Faithfulness checker: verifies every claim in a generated answer against the patient record.

Two layers:
  Layer 1, rule-based (cheap, deterministic, runs on everything):
    - Every cited source ID must exist in this patient's record.
    - Every number in a sentence must appear in the facts that sentence cites.
  Layer 2, LLM-as-a-Judge (a different model family from the generator):
    - Split the output into atomic claims.
    - Label each claim SUPPORTED / UNSUPPORTED / CONTRADICTED, with evidence IDs.

Faithfulness score = supported claims / total claims.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from langchain_groq import ChatGroq

from config import JUDGE_MODEL
from fhir_parser import load_patient, record_as_text

CITATION = re.compile(r"\[([A-Za-z]+/[A-Za-z0-9-]+)\]")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER = re.compile(r"(?<![A-Za-z0-9/-])\d+(?:\.\d+)?")

# LLMs often write Unicode dashes and narrow spaces ("2026‑04‑20", "9.1 %").
# Normalize them so the same value is recognized in both texts.
_UNICODE_FIXES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                                "—": "-", "−": "-", " ": " ", " ": " "})


def _normalize(text: str) -> str:
    return text.translate(_UNICODE_FIXES)


@dataclass
class RuleIssue:
    kind: str       # invalid_citation | number_not_in_source | date_not_in_source | no_citation
    detail: str


@dataclass
class Claim:
    claim: str
    verdict: str                # SUPPORTED | UNSUPPORTED | CONTRADICTED
    evidence: list[str]
    reason: str


@dataclass
class JudgeReport:
    claims: list[Claim] = field(default_factory=list)
    rule_issues: list[RuleIssue] = field(default_factory=list)

    @property
    def faithfulness(self) -> float | None:
        if not self.claims:
            return None
        return sum(c.verdict == "SUPPORTED" for c in self.claims) / len(self.claims)

    @property
    def flagged(self) -> bool:
        """True if a human should review this output."""
        return bool(self.rule_issues) or any(c.verdict != "SUPPORTED" for c in self.claims)


# --- Layer 1: rule-based checks -------------------------------------------------------
def _strip_citations(text: str) -> str:
    return CITATION.sub("", text)


def _ages(patient_fact: str) -> str:
    """Possible current ages for a 'born YYYY-MM-DD' fact (two values, birthday not yet / already passed)."""
    m = re.search(r"born (\d{4})", patient_fact)
    if not m:
        return ""
    age = date.today().year - int(m.group(1))
    return f"{age - 1} {age}"


def rule_checks(output: str, patient_id: str) -> list[RuleIssue]:
    facts = {f.source_id: f.text for f in load_patient(patient_id)}
    issues = []
    for line in _normalize(output).splitlines():
        line = line.strip()
        if not line:
            continue
        cited = CITATION.findall(line)
        for sid in cited:
            if sid not in facts:
                issues.append(RuleIssue("invalid_citation", f"{sid} does not exist in this patient's record"))
        content = _strip_citations(line)
        if (DATE.search(content) or NUMBER.search(content)) and not cited:
            issues.append(RuleIssue("no_citation", f"Statement with values has no citation: {content.strip()[:120]}"))
            continue
        cited_text = " ".join(facts.get(sid, "") for sid in cited)
        # Dates are checked as whole strings, then removed so their parts aren't checked as numbers.
        for d in DATE.findall(content):
            if d not in cited_text:
                issues.append(RuleIssue("date_not_in_source",
                                        f"'{d}' is not in the cited source(s): {content.strip()[:120]}"))
        numbers = NUMBER.findall(DATE.sub(" ", content))
        # Age is a derived value: accept it if it matches the cited birth date.
        cited_text += " " + " ".join(_ages(facts[sid]) for sid in cited if sid.startswith("Patient/") and sid in facts)
        for n in numbers:
            # Compare numerically, so "9.10" matches "9.1".
            if not any(float(n) == float(m) for m in NUMBER.findall(cited_text)):
                issues.append(RuleIssue("number_not_in_source",
                                        f"'{n}' is not in the cited source(s): {content.strip()[:120]}"))
    return issues


# --- Layer 2: LLM-as-a-Judge ----------------------------------------------------------
JUDGE_PROMPT = """You are a strict clinical fact-checker. Check whether a generated text is faithful to a patient record.

Patient record (each fact has a source ID):
{record}

{question_block}Generated text:
\"\"\"
{output}
\"\"\"

Instructions:
1. Split the generated text into atomic factual claims (one fact each). Ignore headings and phrases like "None recorded".
2. For each claim, compare it ONLY with the patient record. Do not use outside medical knowledge.
   - SUPPORTED: the record states this.
   - UNSUPPORTED: the record does not contain this (even if it might be medically true).
   - CONTRADICTED: the record states something different (wrong value, wrong date, wrong status, wrong drug).
3. "Current"/"active" medications must have status active. Calling a stopped medication current is CONTRADICTED.
4. Give the source IDs that support or contradict the claim.

Reply with JSON only, no other text, in this format:
{{"claims": [{{"claim": "...", "verdict": "SUPPORTED", "evidence": ["Observation/p01-obs-1"], "reason": "..."}}]}}"""


@lru_cache
def _judge_llm() -> ChatGroq:
    return ChatGroq(model=JUDGE_MODEL, temperature=0, max_retries=8, reasoning_effort="medium",
                    model_kwargs={"response_format": {"type": "json_object"}})


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


QUESTION_BLOCK = """The generated text is an answer to this physician question: "{question}"
If the answer says "Not found in the patient record", treat it as the claim
"the record does not contain the answer to this question". It is SUPPORTED if the record
really has no such information, and CONTRADICTED if the record does contain it.

"""


def llm_judge(output: str, patient_id: str, question: str | None = None) -> list[Claim]:
    # Without the question, "Not found in the patient record" looks like "the record is empty"
    # and gets wrongly marked CONTRADICTED. The judge needs the same context the generator had.
    record = record_as_text(load_patient(patient_id))
    question_block = QUESTION_BLOCK.format(question=question) if question else ""
    prompt = JUDGE_PROMPT.format(record=record, output=output, question_block=question_block)
    raw = _judge_llm().invoke(prompt).content
    data = _parse_json(raw)
    claims = []
    for c in data.get("claims", []):
        verdict = str(c.get("verdict", "UNSUPPORTED")).upper()
        if verdict not in {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}:
            verdict = "UNSUPPORTED"
        claims.append(Claim(c.get("claim", ""), verdict, c.get("evidence", []) or [], c.get("reason", "")))
    return claims


def evaluate(output: str, patient_id: str, question: str | None = None) -> JudgeReport:
    return JudgeReport(claims=llm_judge(output, patient_id, question), rule_issues=rule_checks(output, patient_id))
