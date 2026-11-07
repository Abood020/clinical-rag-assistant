"""
Vector store: embeds every fact and retrieves the most relevant ones for a question.

Every search is FILTERED by patient_id. A question about one patient can never
retrieve another patient's data. In a real HIE this is an access-control
requirement, not just a relevance trick.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

from config import CHROMA_DIR, EMBED_MODEL, TOP_K
from fhir_parser import Fact, list_patients, load_patient

COLLECTION = "patient_facts"


@lru_cache
def _embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def _to_document(patient_id: str, fact: Fact) -> Document:
    return Document(
        page_content=fact.text,
        metadata={"patient_id": patient_id, "source_id": fact.source_id,
                  "category": fact.category, "date": fact.date},
    )


def build_index() -> Chroma:
    """(Re)build the index from all FHIR bundles."""
    docs, ids = [], []
    for pid in list_patients():
        for fact in load_patient(pid):
            docs.append(_to_document(pid, fact))
            ids.append(fact.source_id)
    store = Chroma(collection_name=COLLECTION, embedding_function=_embeddings(),
                   persist_directory=str(CHROMA_DIR))
    store.reset_collection()
    store.add_documents(docs, ids=ids)
    return store


@lru_cache
def get_store() -> Chroma:
    store = Chroma(collection_name=COLLECTION, embedding_function=_embeddings(),
                   persist_directory=str(CHROMA_DIR))
    if not store.get(limit=1)["ids"]:
        store = build_index()
    return store


def retrieve(patient_id: str, query: str, k: int = TOP_K) -> list[Document]:
    return get_store().similarity_search(query, k=k, filter={"patient_id": patient_id})


if __name__ == "__main__":
    build_index()
    for d in retrieve("p01", "How is the patient's diabetes control?"):
        print(f"[{d.metadata['source_id']}] {d.page_content}")
