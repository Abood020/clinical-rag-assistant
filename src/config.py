"""Central configuration. Swapping a model (e.g. to a self-hosted one) is a one-line change."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"

# Both are open-weight models, so they could be self-hosted on-premise in production.

# Generator: writes summaries and answers (smaller, faster model).
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-oss-20b")

# Judge: checks the generator's output. A larger model than the generator, so the
# checker is more capable than what it checks.
# Known limitation: same model family as the generator, which can share blind spots
# (self-preference bias). Ideally the judge is a different family (e.g. Qwen, Llama);
# that's a one-line change here, but those models hit free-tier rate limits.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

# Small, CPU-friendly embedding model (ONNX via fastembed, 384 dimensions).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Number of facts retrieved per question.
TOP_K = 8
