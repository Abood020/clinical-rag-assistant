"""
The clinical assistant: answers questions (RAG) and writes patient summaries.

Grounding rules given to the model:
  1. Use ONLY the facts provided.
  2. Cite the source ID after every statement, e.g. [Observation/p01-obs-3].
  3. If the answer is not in the facts, say so. Never fill gaps with general knowledge.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_groq import ChatGroq

from config import GENERATOR_MODEL
from fhir_parser import load_patient, record_as_text
from vector_store import retrieve

# Like Malaffi's AI Assistant, clinicians get ready-made questions plus free text.
PREDEFINED_QUERIES = {
    "Active medications": "What medications is the patient currently taking? Include doses.",
    "Abnormal labs": "Which recent lab results or vital signs are outside the reference range?",
    "Chronic conditions": "What active chronic conditions does the patient have, and since when?",
    "Allergies": "Does the patient have any allergies? Include the reaction and severity.",
    "Recent visits": "What were the patient's most recent visits and why?",
}

GROUNDING_RULES = """Rules you must follow:
- Use ONLY the facts below. Do not use general medical knowledge to add information.
- After EVERY statement, cite the source ID(s) in square brackets, exactly as written, e.g. [Observation/p01-obs-3].
- Medications with status "stopped" are NOT current medications.
- If the facts do not contain the answer, reply: "Not found in the patient record."
- Be concise. Use short bullet points."""

ANSWER_PROMPT = """You are a clinical assistant helping a physician review a patient's health record.
{rules}

Facts from the patient record:
{facts}

Physician's question: {question}"""

SUMMARY_PROMPT = """You are a clinical assistant. Write a short clinical summary of this patient for a physician.
{rules}

Structure the summary with these headings: Overview, Active problems, Current medications,
Allergies, Notable results and trends, Recent care. If a section has no data, write "None recorded."
Point out trends in repeated lab results (e.g. rising or falling values).

Full patient record:
{facts}"""


@dataclass
class AssistantOutput:
    text: str
    sources: list[str] = field(default_factory=list)   # facts the model was shown


@lru_cache
def _llm() -> ChatGroq:
    return ChatGroq(model=GENERATOR_MODEL, temperature=0, max_retries=8, reasoning_effort="low")


def answer(patient_id: str, question: str) -> AssistantOutput:
    """RAG: retrieve the most relevant facts for this patient, then answer from them only."""
    docs = retrieve(patient_id, question)
    facts = "\n".join(f"[{d.metadata['source_id']}] {d.page_content}" for d in docs)
    prompt = ANSWER_PROMPT.format(rules=GROUNDING_RULES, facts=facts, question=question)
    return AssistantOutput(_llm().invoke(prompt).content, facts.splitlines())


def summarize(patient_id: str) -> AssistantOutput:
    """
    A summary needs the WHOLE record, not the top-k matches, so it does not use retrieval.
    These synthetic records are small. A real record with years of data would need
    a map-reduce approach: summarize each section, then combine the section summaries.
    """
    facts = record_as_text(load_patient(patient_id))
    prompt = SUMMARY_PROMPT.format(rules=GROUNDING_RULES, facts=facts)
    return AssistantOutput(_llm().invoke(prompt).content, facts.splitlines())
