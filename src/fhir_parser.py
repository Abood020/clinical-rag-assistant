"""
Turn a FHIR Bundle into a list of plain-text "facts".

Chunking strategy: ONE FHIR resource = ONE chunk.
Clinical records are already structured, so each resource (a lab result,
a medication, a diagnosis) is a natural, self-contained unit of meaning.
Keeping the resource ID with each fact lets every statement in an answer
be traced back to its exact source (citations).
"""

import json
from dataclasses import dataclass
from pathlib import Path

FHIR_DIR = Path(__file__).resolve().parent.parent / "data" / "fhir"


@dataclass
class Fact:
    source_id: str      # e.g. "Observation/p01-obs-3"
    category: str       # condition | medication | allergy | lab | encounter | demographics
    text: str           # human-readable sentence used for retrieval and by the LLM
    date: str = ""      # used to sort facts chronologically


def _first_text(codeable: dict) -> str:
    if not codeable:
        return ""
    if codeable.get("text"):
        return codeable["text"]
    codings = codeable.get("coding") or [{}]
    return codings[0].get("display", "")


def _status(resource: dict, field: str) -> str:
    value = resource.get(field, "")
    if isinstance(value, dict):
        return (value.get("coding") or [{}])[0].get("code", "")
    return value


INTERPRETATION = {"H": "HIGH", "L": "LOW", "N": "normal"}


def resource_to_fact(r: dict) -> Fact | None:
    rtype, rid = r["resourceType"], r["id"]
    source_id = f"{rtype}/{rid}"

    if rtype == "Patient":
        name = r["name"][0]
        full = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
        return Fact(source_id, "demographics",
                    f"Patient {full}, {r.get('gender')}, born {r.get('birthDate')}.")

    if rtype == "Condition":
        status = _status(r, "clinicalStatus")
        return Fact(source_id, "condition",
                    f"Diagnosis: {_first_text(r['code'])} (status: {status}), onset {r.get('onsetDateTime')}.",
                    r.get("onsetDateTime", ""))

    if rtype == "MedicationRequest":
        dose = (r.get("dosageInstruction") or [{}])[0].get("text", "")
        return Fact(source_id, "medication",
                    f"Medication: {_first_text(r['medicationCodeableConcept'])}, {dose} "
                    f"(status: {r.get('status')}), started {r.get('authoredOn')}.",
                    r.get("authoredOn", ""))

    if rtype == "AllergyIntolerance":
        reaction = r["reaction"][0]["manifestation"][0]["text"] if r.get("reaction") else "unknown reaction"
        return Fact(source_id, "allergy",
                    f"Allergy: {_first_text(r['code'])}, reaction: {reaction} "
                    f"(criticality: {r.get('criticality')}).")

    if rtype == "Observation":
        q = r["valueQuantity"]
        rng = r.get("referenceRange", [{}])[0]
        interp = (r.get("interpretation") or [{}])[0].get("coding", [{}])[0].get("code", "")
        flag = INTERPRETATION.get(interp, "")
        return Fact(source_id, "lab",
                    f"Result: {_first_text(r['code'])} = {q['value']} {q.get('unit', '')} "
                    f"on {r.get('effectiveDateTime')} "
                    f"(reference range {rng.get('low', {}).get('value')}-{rng.get('high', {}).get('value')}, "
                    f"{flag}).",
                    r.get("effectiveDateTime", ""))

    if rtype == "Encounter":
        reason = (r.get("reasonCode") or [{}])[0].get("text", "")
        loc = (r.get("location") or [{}])[0].get("location", {}).get("display", "")
        start = r.get("period", {}).get("start", "")
        return Fact(source_id, "encounter", f"Visit on {start}: {reason} at {loc}.", start)

    return None


def load_patient(patient_id: str) -> list[Fact]:
    bundle = json.loads((FHIR_DIR / f"{patient_id}.json").read_text())
    facts = [resource_to_fact(e["resource"]) for e in bundle["entry"]]
    return [f for f in facts if f is not None]


def list_patients() -> dict[str, str]:
    """Return {patient_id: display name}."""
    out = {}
    for path in sorted(FHIR_DIR.glob("*.json")):
        facts = load_patient(path.stem)
        demo = next(f for f in facts if f.category == "demographics")
        out[path.stem] = demo.text.removeprefix("Patient ").split(",")[0]
    return out


def record_as_text(facts: list[Fact]) -> str:
    """Full record, one fact per line, each tagged with its source ID."""
    order = ["demographics", "condition", "medication", "allergy", "lab", "encounter"]
    facts = sorted(facts, key=lambda f: (order.index(f.category), f.date))
    return "\n".join(f"[{f.source_id}] {f.text}" for f in facts)
