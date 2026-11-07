"""
Evaluation of the faithfulness checker.

Question: "How do we know the judge is reliable?"
Answer: test it on outputs where we KNOW the ground truth.

Method (error injection):
  1. Generate a clinical summary for every patient (the clean version).
  2. Create corrupted copies, each with ONE known error of a specific type.
  3. Run the checker on the clean and corrupted versions.
  4. Measure:
     - Recall per error type: how many injected errors did each layer catch?
     - False-alarm rate: how often was a clean summary flagged?

Also runs a set of "trap" questions (answers that are NOT in the record) to test
whether the assistant itself refuses to invent information.

Usage:  python eval/run_eval.py
Output: eval/results.json and a printed report.
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant import answer, summarize          # noqa: E402
from fhir_parser import list_patients, load_patient  # noqa: E402
from judge import evaluate                       # noqa: E402

OUT = Path(__file__).parent / "results.json"


# --- Error injection -------------------------------------------------------------------
def inject_wrong_value(summary: str, pid: str) -> str | None:
    """Change one lab value that is cited from an Observation."""
    for line in summary.splitlines():
        if f"[Observation/{pid}-obs-" in line:
            m = re.search(r"(?<![\d.\-‑])(\d+(?:\.\d+)?)\s? ?(%|mg|mmol|mL|g/dL|ng|mIU|pg|mmHg|kg)", line)
            if m:
                wrong = round(float(m.group(1)) * 1.5 + 1, 1)
                return summary.replace(line, line.replace(m.group(1), str(wrong), 1), 1)
    return None


def inject_fabricated_med(summary: str, pid: str) -> str | None:
    """Add a medication that is not in the record, with a plausible-looking citation."""
    meds = [f.source_id for f in load_patient(pid) if f.category == "medication"]
    cite = meds[0] if meds else f"Patient/{pid}"
    return summary + f"\n- Atenolol 50 mg tablet, 1 tablet once daily [{cite}]"


def inject_unsupported_claim(summary: str, pid: str) -> str | None:
    """Add a claim that sounds clinical but is not stated anywhere in the record."""
    conds = [f.source_id for f in load_patient(pid) if f.category == "condition"]
    if not conds:
        return None
    return summary + f"\n- The patient was referred for bariatric surgery evaluation [{conds[0]}]"


def inject_contradiction(summary: str, pid: str) -> str | None:
    """Deny an allergy that IS in the record."""
    if not any(f.category == "allergy" for f in load_patient(pid)):
        return None
    return summary + f"\n- No known drug or food allergies [Patient/{pid}]"


INJECTIONS = {
    "wrong_value": inject_wrong_value,
    "fabricated_medication": inject_fabricated_med,
    "unsupported_claim": inject_unsupported_claim,
    "contradicted_allergy": inject_contradiction,
}

# --- Trap questions: the correct answer is "not in the record" ------------------------
TRAP_QUESTIONS = [
    ("p05", "What diabetes medication is the patient taking?"),
    ("p10", "What chronic conditions does the patient have?"),
    ("p06", "What was the patient's most recent HbA1c?"),
    ("p08", "What is the patient's blood pressure?"),
]


def non_supported(report) -> int:
    return sum(c.verdict != "SUPPORTED" for c in report.claims)


def main() -> None:
    results = {"injection": [], "traps": []}
    caught = defaultdict(lambda: {"rules": 0, "judge": 0, "either": 0, "total": 0})
    clean_flags = {"rules": 0, "judge": 0, "total": 0}

    for pid in list_patients():
        print(f"Patient {pid}...")
        summary = summarize(pid).text
        clean = evaluate(summary, pid)
        clean_flags["total"] += 1
        clean_flags["rules"] += bool(clean.rule_issues)
        clean_flags["judge"] += non_supported(clean) > 0
        results["injection"].append({"patient": pid, "type": "clean", "faithfulness": clean.faithfulness,
                                     "rule_issues": len(clean.rule_issues), "non_supported": non_supported(clean)})

        for name, inject in INJECTIONS.items():
            corrupted = inject(summary, pid)
            if corrupted is None:
                continue
            report = evaluate(corrupted, pid)
            # An error counts as caught if the layer reports MORE problems than on the clean version.
            by_rules = len(report.rule_issues) > len(clean.rule_issues)
            by_judge = non_supported(report) > non_supported(clean)
            stats = caught[name]
            stats["total"] += 1
            stats["rules"] += by_rules
            stats["judge"] += by_judge
            stats["either"] += by_rules or by_judge
            results["injection"].append({"patient": pid, "type": name, "faithfulness": report.faithfulness,
                                         "caught_by_rules": by_rules, "caught_by_judge": by_judge})
            time.sleep(8)  # stay within free-tier token-per-minute limits

    for pid, q in TRAP_QUESTIONS:
        out = answer(pid, q).text
        refused = "not found in the patient record" in out.lower()
        results["traps"].append({"patient": pid, "question": q, "answer": out, "refused_correctly": refused})

    # --- Report ---
    lines = ["\n=== Faithfulness checker: error-injection results ===",
             f"{'Error type':<24}{'Rules':>8}{'Judge':>8}{'Combined':>10}"]
    for name, s in caught.items():
        t = s["total"]
        lines.append(f"{name:<24}{s['rules']}/{t:<6}{s['judge']}/{t:<6}{s['either']}/{t}")
    total = sum(s["total"] for s in caught.values())
    lines.append(f"{'ALL ERRORS':<24}{sum(s['rules'] for s in caught.values())}/{total:<6}"
                 f"{sum(s['judge'] for s in caught.values())}/{total:<6}"
                 f"{sum(s['either'] for s in caught.values())}/{total}")
    t = clean_flags["total"]
    lines.append(f"\nClean summaries flagged (false alarms): rules {clean_flags['rules']}/{t}, "
                 f"judge {clean_flags['judge']}/{t}")
    lines.append(f"Trap questions answered 'not found': "
                 f"{sum(r['refused_correctly'] for r in results['traps'])}/{len(results['traps'])}")
    report = "\n".join(lines)
    print(report)

    results["summary"] = {"caught": caught, "clean_flags": clean_flags, "report": report}
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
