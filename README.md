# Clinical Assistant with a Built-in Faithfulness Checker

A small prototype of an AI clinical assistant over a health information exchange (HIE):
a physician picks a patient, gets an **AI-generated clinical summary**, and can **ask questions
about the patient record**. Every output is automatically **checked claim by claim against the
source record** before a clinician relies on it.

**🔗 Live demo: [clinical-assistant-abdallah.streamlit.app](https://clinical-assistant-abdallah.streamlit.app/)**
Try it: pick *Omar Haddad (p05)*, a diabetic patient with no medication on record, and ask
"What diabetes medication is the patient taking?" The assistant should answer that it is not in the record.

> All patient data is **synthetic**. No real patient information is used.

## Why I built it

AI summaries of patient records are useful only if clinicians can trust them. A summary that
invents a medication, misses an allergy, or copies a wrong lab value is worse than no summary.
My background is in evaluating LLM outputs for faithfulness and hallucinations, so I wanted to
explore that problem in a clinical setting: **how do you make generated clinical text verifiable,
and how do you measure whether your checker actually works?**

## What it does

| Feature | How it works |
|---|---|
| **FHIR patient records** | Synthetic patients stored as FHIR R4 Bundles (Patient, Condition, MedicationRequest, AllergyIntolerance, Observation, Encounter), with SNOMED CT and LOINC codes. |
| **Ask the record (RAG)** | Predefined clinician questions (active medications, abnormal labs, ...) plus free text. Relevant facts are retrieved from a vector store, and the model answers **only** from them, citing the source ID of every statement. |
| **Clinical summary** | Structured summary (problems, medications, allergies, trends in lab results, recent care), with citations. |
| **Faithfulness checker** | Two layers: deterministic rule checks, then an LLM judge that labels each claim *supported / unsupported / contradicted*. Outputs a faithfulness score and flags outputs that need human review. |
| **Evaluation** | Error-injection test that measures how many known errors the checker catches, plus "trap" questions whose correct answer is *not in the record*. |

## Architecture

```
FHIR Bundles ──► fhir_parser ──► one fact per resource, tagged with its source ID
                                      │
                                      ▼
                       embeddings (bge-small, ONNX, CPU) ──► Chroma vector store
                                                                   │  (filtered by patient_id)
Physician question ──► retrieve top-k facts ───────────────────────┘
                                      │
                                      ▼
                       Generator LLM: answer ONLY from facts, cite every statement
                                      │
                                      ▼
            Faithfulness checker ─┬─ Layer 1: rule checks (cited IDs exist, numbers/dates match the cited fact)
                                  └─ Layer 2: LLM judge (claim extraction + verdict + evidence)
                                      │
                                      ▼
                       Faithfulness score · claims table · "needs human review" flag
```

## Design decisions

- **One FHIR resource = one chunk.** Clinical records are already structured; a lab result or a
  medication order is a natural self-contained unit. It also makes precise citations possible.
- **Retrieval is always filtered by `patient_id`.** A question about one patient can never retrieve
  another patient's data. In a real HIE that is an access-control requirement.
- **Citations are mandatory.** If every statement carries a source ID, verification becomes a
  lookup instead of a guess, and a clinician can click through to the source.
- **"Not found in the record" is a valid answer.** The model is told never to fill gaps with general
  medical knowledge. A medically true sentence that is not in the record is still unfaithful.
- **Stopped medications are not current medications.** A common, dangerous error; it is stated in
  the prompt and checked by the judge.
- **Summaries use the full record, not top-k retrieval.** A summary needs everything. For large real
  records this would become map-reduce (summarize each section, then combine).
- **Layered checking.** Cheap deterministic rules catch fabricated citations and wrong numbers on
  100% of outputs; the LLM judge catches meaning-level errors (wrong status, unsupported claims).
  In production, flagged outputs and a random sample of passing ones would go to clinical reviewers.
- **Open-weight models.** Both models are open-weight (`gpt-oss`), so in production they could be
  self-hosted on-premise and patient data would never leave the environment. Models are set in
  `src/config.py`; swapping one is a one-line change.

## Evaluation

Run `python eval/run_eval.py`. For every patient it generates a summary, creates corrupted copies with
**one known error each**, and checks whether the checker catches it:

| Injected error | Example |
|---|---|
| `wrong_value` | a lab value changed (e.g. HbA1c 9.1 → 14.7) |
| `fabricated_medication` | a drug that is not in the record, with a real-looking citation |
| `unsupported_claim` | a plausible clinical statement that appears nowhere in the record |
| `contradicted_allergy` | "No known allergies" for a patient who has a documented allergy |

Results (see `eval/results.json`):

| Injected error | Rules | LLM judge | Combined |
|---|---|---|---|
| wrong_value | 10/10 | 9/10 | **10/10** |
| fabricated_medication | 10/10 | 8/10 | **10/10** |
| unsupported_claim | 0/9 | 8/9 | **8/9** |
| contradicted_allergy | 0/4 | 4/4 | **4/4** |
| **All errors** | 20/33 | 29/33 | **32/33 (97%)** |

- **Trap questions** (answer is not in the record, e.g. "What diabetes medication is the patient taking?"
  for a diabetic patient with no medication on record): **4/4** answered "Not found in the patient record".
- **False alarms on clean summaries:** rules 7/10, judge 3/10.

**What the numbers show:**
- **The two layers are complementary.** Rules catch every wrong number and fake citation but cannot see
  meaning (0% on unsupported claims and contradictions). The judge catches meaning-level errors but
  missed some numeric ones. Together they caught 32 of 33.
- **The rule layer has low precision.** Most of its false alarms were *derived* values that are correct
  but not written verbatim in the record: a patient's age (computed from birth date) or "over ~6 months".
  Age is now handled; durations are not. This is the classic trade-off: a strict layer gives high recall
  and more alarms for human review.
- **The one miss** was an unsupported claim (a referral) that the judge accepted. In a clinical setting,
  misses matter more than false alarms, which is why flagged *and* randomly sampled outputs should go to
  human review.
- Small sample (10 patients, 33 injected errors), so these numbers show the method rather than prove
  production performance.

## Limitations and next steps

- **Small synthetic dataset** (10 patients). Real records are larger, messier, and contradictory.
- **Judge and generator share a model family**, which can share blind spots. A different-family judge
  (and validating the judge against clinician labels, with inter-annotator agreement) is the next step.
- **Injected errors are simple.** Real hallucinations are subtler: wrong timelines, merged events,
  overconfident interpretations.
- **No clinical validation.** Before any real use, clinicians would need to label a gold set, and the
  key metric would be the judge's **recall on dangerous errors**.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=...            # free key from console.groq.com
python data/generate_patients.py   # writes data/fhir/*.json
streamlit run app.py
```

## Project structure

```
data/generate_patients.py   synthetic FHIR R4 bundles (with hallucination "traps")
src/fhir_parser.py          FHIR resources -> cited facts
src/vector_store.py         embeddings + Chroma, patient-filtered retrieval
src/assistant.py            RAG answers and clinical summaries with citations
src/judge.py                faithfulness checker (rules + LLM judge)
eval/run_eval.py            error-injection evaluation
app.py                      Streamlit UI
```
