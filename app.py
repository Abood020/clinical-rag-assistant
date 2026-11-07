"""
Streamlit UI: a mini clinical assistant with a built-in faithfulness checker.
Run:  streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

# On Streamlit Cloud, the API key comes from the app's secrets.
try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass  # running locally: the key comes from the environment variable

from assistant import PREDEFINED_QUERIES, answer, summarize  # noqa: E402
from config import GENERATOR_MODEL, JUDGE_MODEL              # noqa: E402
from fhir_parser import list_patients, load_patient, record_as_text  # noqa: E402
from judge import evaluate                                   # noqa: E402

st.set_page_config(page_title="Clinical Assistant Prototype", page_icon="🩺", layout="wide")

VERDICT_STYLE = {"SUPPORTED": "✅", "UNSUPPORTED": "❓", "CONTRADICTED": "❌"}


def show_check(output_text: str, patient_id: str) -> None:
    with st.spinner("Checking every claim against the patient record..."):
        report = evaluate(output_text, patient_id)

    score = report.faithfulness
    c1, c2, c3 = st.columns(3)
    c1.metric("Faithfulness", "n/a" if score is None else f"{score:.0%}")
    c2.metric("Claims checked", len(report.claims))
    c3.metric("Needs human review", "Yes" if report.flagged else "No")

    if report.rule_issues:
        st.warning("**Rule-based checks found issues:**\n\n" +
                   "\n".join(f"- `{i.kind}`: {i.detail}" for i in report.rule_issues))

    with st.expander("Claim-by-claim verdicts (LLM judge)", expanded=report.flagged):
        for c in report.claims:
            evidence = ", ".join(c.evidence) if c.evidence else "none"
            st.markdown(f"{VERDICT_STYLE.get(c.verdict, '•')} **{c.verdict}**: {c.claim}  \n"
                        f"<small>Evidence: {evidence}. {c.reason}</small>", unsafe_allow_html=True)


# --- Sidebar ---
patients = list_patients()
with st.sidebar:
    st.title("🩺 Clinical Assistant")
    st.caption("Prototype · synthetic FHIR data only")
    pid = st.selectbox("Patient", list(patients), format_func=lambda p: f"{patients[p]} ({p})")
    st.divider()
    st.markdown(f"**Generator:** `{GENERATOR_MODEL}`  \n**Judge:** `{JUDGE_MODEL}`")
    st.caption("The judge is a larger model than the generator, so the checker is more capable "
               "than what it checks. Both are open-weight and could be self-hosted.")

st.header(patients[pid])
tab_summary, tab_ask, tab_record = st.tabs(["📋 Clinical summary", "💬 Ask the record", "🗂️ Source record (FHIR)"])

with tab_summary:
    if st.button("Generate summary", type="primary"):
        with st.spinner("Writing summary..."):
            st.session_state[f"summary_{pid}"] = summarize(pid).text
    if text := st.session_state.get(f"summary_{pid}"):
        st.markdown(text)
        st.subheader("Faithfulness check")
        show_check(text, pid)

with tab_ask:
    st.write("Quick questions:")
    cols = st.columns(len(PREDEFINED_QUERIES))
    question = None
    for col, (label, q) in zip(cols, PREDEFINED_QUERIES.items()):
        if col.button(label, use_container_width=True):
            question = q
    free = st.chat_input("Ask anything about this patient...")
    question = free or question
    if question:
        st.markdown(f"**Question:** {question}")
        with st.spinner("Retrieving facts and answering..."):
            out = answer(pid, question)
        st.markdown(out.text)
        with st.expander("Facts retrieved for this question (RAG context)"):
            st.code("\n".join(out.sources), language=None)
        st.subheader("Faithfulness check")
        show_check(out.text, pid)

with tab_record:
    st.caption("Each line is one FHIR resource. The ID in brackets is what the assistant cites.")
    st.code(record_as_text(load_patient(pid)), language=None)
