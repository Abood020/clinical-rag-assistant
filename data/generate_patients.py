"""
Generate synthetic patient records as FHIR R4 Bundles.

All data is fictional. Each patient profile is designed around a realistic
clinical scenario, and a few include deliberate "traps" for hallucination testing
(e.g. a diabetic patient with NO diabetes medication on record, or a patient
with an almost empty record).

Output: data/fhir/<patient_id>.json  (one FHIR Bundle per patient)
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "fhir"

# --- Standard codes (subset) -------------------------------------------------
# LOINC codes for lab tests / vitals, with unit and reference range.
LABS = {
    "hba1c":       ("4548-4",  "Hemoglobin A1c",              "%",        4.0, 5.6),
    "glucose":     ("2345-7",  "Fasting glucose",             "mg/dL",    70, 99),
    "creatinine":  ("2160-0",  "Creatinine",                  "mg/dL",    0.6, 1.2),
    "egfr":        ("33914-3", "eGFR",                        "mL/min/1.73m2", 90, 120),
    "ldl":         ("18262-6", "LDL cholesterol",             "mg/dL",    0, 100),
    "potassium":   ("2823-3",  "Potassium",                   "mmol/L",   3.5, 5.1),
    "hemoglobin":  ("718-7",   "Hemoglobin",                  "g/dL",     12.0, 17.5),
    "ferritin":    ("2276-4",  "Ferritin",                    "ng/mL",    30, 400),
    "tsh":         ("3016-3",  "TSH",                         "mIU/L",    0.4, 4.0),
    "inr":         ("6301-6",  "INR",                         "",         2.0, 3.0),
    "bnp":         ("30934-4", "BNP",                         "pg/mL",    0, 100),
    "troponin":    ("6598-7",  "Troponin T",                  "ng/mL",    0, 0.01),
    "sbp":         ("8480-6",  "Systolic blood pressure",     "mmHg",     90, 130),
    "dbp":         ("8462-4",  "Diastolic blood pressure",    "mmHg",     60, 85),
    "bmi":         ("39156-5", "Body mass index",             "kg/m2",    18.5, 24.9),
    "spo2":        ("59408-5", "Oxygen saturation",           "%",        95, 100),
}

# --- Patient profiles ----------------------------------------------------------
# labs: list of (lab_key, value, date)
# meds: list of (name, dose_instruction, status, start_date)
PATIENTS = [
    {
        "id": "p01", "name": ("Khalid", "Al Mansoori"), "gender": "male", "birth": "1968-03-14",
        "conditions": [("44054006", "Type 2 diabetes mellitus", "2016-05-02", "active"),
                       ("38341003", "Essential hypertension", "2018-09-11", "active")],
        "meds": [("Metformin 1000 mg tablet", "1 tablet twice daily", "active", "2016-05-02"),
                 ("Lisinopril 10 mg tablet", "1 tablet once daily", "active", "2018-09-11")],
        "allergies": [("Penicillin", "Hives", "high")],
        "labs": [("hba1c", 7.8, "2025-04-10"), ("hba1c", 8.4, "2025-10-15"), ("hba1c", 9.1, "2026-04-20"),
                 ("glucose", 168, "2026-04-20"), ("sbp", 142, "2026-04-20"), ("dbp", 88, "2026-04-20"),
                 ("bmi", 31.2, "2026-04-20"), ("creatinine", 1.0, "2026-04-20")],
        "encounters": [("2026-04-20", "Diabetes follow-up visit", "Endocrinology clinic")],
    },
    {
        "id": "p02", "name": ("Fatima", "Al Hosani"), "gender": "female", "birth": "1955-07-22",
        "conditions": [("44054006", "Type 2 diabetes mellitus", "2010-02-18", "active"),
                       ("433144002", "Chronic kidney disease stage 3", "2023-06-05", "active")],
        "meds": [("Metformin 500 mg tablet", "1 tablet once daily (dose reduced for kidney function)", "active", "2023-06-05"),
                 ("Empagliflozin 10 mg tablet", "1 tablet once daily", "active", "2024-01-12"),
                 ("Metformin 1000 mg tablet", "1 tablet twice daily", "stopped", "2010-02-18")],
        "allergies": [],
        "labs": [("egfr", 58, "2024-05-01"), ("egfr", 51, "2025-05-03"), ("egfr", 44, "2026-05-06"),
                 ("creatinine", 1.4, "2026-05-06"), ("potassium", 5.4, "2026-05-06"), ("hba1c", 7.1, "2026-05-06")],
        "encounters": [("2026-05-06", "Nephrology review", "Nephrology clinic")],
    },
    {
        "id": "p03", "name": ("Rajesh", "Nair"), "gender": "male", "birth": "1950-11-03",
        "conditions": [("84114007", "Heart failure", "2021-03-19", "active"),
                       ("49436004", "Atrial fibrillation", "2021-03-19", "active")],
        "meds": [("Warfarin 5 mg tablet", "Dose adjusted per INR", "active", "2021-03-20"),
                 ("Furosemide 40 mg tablet", "1 tablet once daily", "active", "2021-03-20"),
                 ("Bisoprolol 5 mg tablet", "1 tablet once daily", "active", "2021-03-20")],
        "allergies": [("Sulfonamide antibiotics", "Rash", "low")],
        "labs": [("inr", 1.6, "2026-06-01"), ("inr", 2.4, "2026-07-01"), ("bnp", 480, "2026-07-01"),
                 ("potassium", 3.9, "2026-07-01"), ("sbp", 118, "2026-07-01"), ("dbp", 72, "2026-07-01")],
        "encounters": [("2026-06-28", "Emergency visit: shortness of breath", "Emergency department"),
                       ("2026-07-01", "Anticoagulation clinic", "Cardiology clinic")],
    },
    {
        "id": "p04", "name": ("Maryam", "Al Ketbi"), "gender": "female", "birth": "1994-01-30",
        "conditions": [("11687002", "Gestational diabetes", "2026-08-14", "active"),
                       ("77386006", "Pregnancy", "2026-03-01", "active")],
        "meds": [("Insulin aspart", "Per sliding scale with meals", "active", "2026-08-20"),
                 ("Folic acid 400 mcg tablet", "1 tablet once daily", "active", "2026-03-10")],
        "allergies": [],
        "labs": [("glucose", 112, "2026-08-14"), ("glucose", 104, "2026-09-20"),
                 ("sbp", 136, "2026-09-20"), ("dbp", 87, "2026-09-20"), ("hemoglobin", 10.9, "2026-09-20")],
        "encounters": [("2026-09-20", "Antenatal visit (28 weeks)", "Obstetrics clinic")],
    },
    {
        "id": "p05", "name": ("Omar", "Haddad"), "gender": "male", "birth": "1972-09-09",
        # TRAP: diabetes on the problem list, but NO diabetes medication recorded.
        "conditions": [("44054006", "Type 2 diabetes mellitus", "2025-11-02", "active")],
        "meds": [],
        "allergies": [("Shellfish", "Anaphylaxis", "high")],
        "labs": [("hba1c", 6.9, "2025-11-02"), ("hba1c", 7.2, "2026-05-10"), ("bmi", 29.4, "2026-05-10")],
        "encounters": [("2026-05-10", "Primary care visit: lifestyle counselling", "Family medicine clinic")],
    },
    {
        "id": "p06", "name": ("Aisha", "Al Shamsi"), "gender": "female", "birth": "1980-04-17",
        "conditions": [("40930008", "Hypothyroidism", "2019-08-08", "active")],
        "meds": [("Levothyroxine 75 mcg tablet", "1 tablet every morning", "active", "2019-08-08")],
        "allergies": [],
        "labs": [("tsh", 6.8, "2025-12-01"), ("tsh", 3.1, "2026-06-03")],
        "encounters": [("2026-06-03", "Thyroid follow-up", "Endocrinology clinic")],
    },
    {
        "id": "p07", "name": ("Hassan", "Al Zaabi"), "gender": "male", "birth": "1960-12-25",
        "conditions": [("22298006", "Myocardial infarction", "2025-02-11", "resolved"),
                       ("53741008", "Coronary artery disease", "2025-02-11", "active"),
                       ("55822004", "Hyperlipidemia", "2017-04-04", "active")],
        "meds": [("Aspirin 81 mg tablet", "1 tablet once daily", "active", "2025-02-12"),
                 ("Clopidogrel 75 mg tablet", "1 tablet once daily", "stopped", "2025-02-12"),
                 ("Atorvastatin 80 mg tablet", "1 tablet at night", "active", "2025-02-12"),
                 ("Metoprolol 50 mg tablet", "1 tablet twice daily", "active", "2025-02-12")],
        "allergies": [],
        "labs": [("troponin", 2.3, "2025-02-11"), ("ldl", 162, "2025-02-11"), ("ldl", 88, "2025-08-15"),
                 ("ldl", 64, "2026-03-01")],
        "encounters": [("2025-02-11", "Admission: acute myocardial infarction", "Cardiac care unit"),
                       ("2026-03-01", "Cardiology follow-up", "Cardiology clinic")],
    },
    {
        "id": "p08", "name": ("Sara", "Mathews"), "gender": "female", "birth": "1988-06-02",
        "conditions": [("87522002", "Iron deficiency anemia", "2026-02-15", "active")],
        "meds": [("Ferrous sulfate 325 mg tablet", "1 tablet once daily", "active", "2026-02-15")],
        "allergies": [("Ibuprofen", "Bronchospasm", "high")],
        "labs": [("hemoglobin", 9.4, "2026-02-15"), ("ferritin", 8, "2026-02-15"),
                 ("hemoglobin", 11.6, "2026-05-20"), ("ferritin", 24, "2026-05-20")],
        "encounters": [("2026-05-20", "Anemia follow-up", "Internal medicine clinic")],
    },
    {
        "id": "p09", "name": ("Yousef", "Al Dhaheri"), "gender": "male", "birth": "1958-10-10",
        "conditions": [("13645005", "Chronic obstructive pulmonary disease", "2018-01-22", "active")],
        "meds": [("Tiotropium inhaler 18 mcg", "1 inhalation once daily", "active", "2018-01-22"),
                 ("Salbutamol inhaler 100 mcg", "2 puffs as needed", "active", "2018-01-22")],
        "allergies": [],
        "labs": [("spo2", 91, "2026-01-14"), ("spo2", 94, "2026-03-02")],
        "encounters": [("2026-01-14", "Hospital admission: COPD exacerbation", "Pulmonology ward"),
                       ("2026-03-02", "Pulmonology follow-up", "Pulmonology clinic")],
    },
    {
        "id": "p10", "name": ("Noura", "Al Suwaidi"), "gender": "female", "birth": "1999-02-08",
        # TRAP: almost empty record. A good summary should say little, not invent history.
        "conditions": [],
        "meds": [],
        "allergies": [],
        "labs": [("hemoglobin", 13.5, "2026-01-05")],
        "encounters": [("2026-01-05", "Pre-employment medical check", "Family medicine clinic")],
    },
]


# --- FHIR resource builders -------------------------------------------------------
def ref(resource_type: str, rid: str) -> dict:
    return {"reference": f"{resource_type}/{rid}"}


def interpretation(value: float, low: float, high: float) -> str:
    if value < low:
        return "L"
    if value > high:
        return "H"
    return "N"


def build_bundle(p: dict) -> dict:
    pid = p["id"]
    entries = []

    def add(resource: dict) -> None:
        entries.append({"fullUrl": f"urn:uuid:{resource['id']}", "resource": resource})

    add({
        "resourceType": "Patient", "id": pid,
        "name": [{"given": [p["name"][0]], "family": p["name"][1]}],
        "gender": p["gender"], "birthDate": p["birth"],
    })

    for i, (code, display, onset, status) in enumerate(p["conditions"], 1):
        add({
            "resourceType": "Condition", "id": f"{pid}-cond-{i}",
            "subject": ref("Patient", pid),
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}],
                     "text": display},
            "clinicalStatus": {"coding": [{"code": status}]},
            "onsetDateTime": onset,
        })

    for i, (name, dose, status, start) in enumerate(p["meds"], 1):
        add({
            "resourceType": "MedicationRequest", "id": f"{pid}-med-{i}",
            "subject": ref("Patient", pid), "status": status, "intent": "order",
            "medicationCodeableConcept": {"text": name},
            "dosageInstruction": [{"text": dose}],
            "authoredOn": start,
        })

    for i, (substance, reaction, criticality) in enumerate(p["allergies"], 1):
        add({
            "resourceType": "AllergyIntolerance", "id": f"{pid}-allergy-{i}",
            "patient": ref("Patient", pid), "criticality": criticality,
            "code": {"text": substance},
            "reaction": [{"manifestation": [{"text": reaction}]}],
        })

    for i, (key, value, date) in enumerate(p["labs"], 1):
        loinc, display, unit, low, high = LABS[key]
        add({
            "resourceType": "Observation", "id": f"{pid}-obs-{i}", "status": "final",
            "subject": ref("Patient", pid),
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}],
                     "text": display},
            "effectiveDateTime": date,
            "valueQuantity": {"value": value, "unit": unit},
            "referenceRange": [{"low": {"value": low}, "high": {"value": high}}],
            "interpretation": [{"coding": [{"code": interpretation(value, low, high)}]}],
        })

    for i, (date, reason, location) in enumerate(p["encounters"], 1):
        add({
            "resourceType": "Encounter", "id": f"{pid}-enc-{i}", "status": "finished",
            "subject": ref("Patient", pid),
            "period": {"start": date},
            "reasonCode": [{"text": reason}],
            "location": [{"location": {"display": location}}],
        })

    return {"resourceType": "Bundle", "type": "collection", "id": f"bundle-{pid}", "entry": entries}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in PATIENTS:
        path = OUT_DIR / f"{p['id']}.json"
        path.write_text(json.dumps(build_bundle(p), indent=2))
    print(f"Wrote {len(PATIENTS)} FHIR bundles to {OUT_DIR}")


if __name__ == "__main__":
    main()
