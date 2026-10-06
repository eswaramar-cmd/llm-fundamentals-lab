"""Confirm the RAG store still resolves to the populated index."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from backend.agent.config import PROJECT_ROOT, get_settings  # noqa: E402

s = get_settings()

print(f"  chroma_rag   : {s.chroma_rag_path}")
print(f"  under root   : {s.chroma_rag_path.startswith(str(PROJECT_ROOT))}")
print(f"  under backend: {str(PROJECT_ROOT / 'backend') in s.chroma_rag_path}")
print(f"  exists       : {Path(s.chroma_rag_path).exists()}")
print()

from backend.agent.tools.knowledge_base import _get_vectorstore  # noqa: E402

vs = _get_vectorstore()
docs = vs.similarity_search("Apex Care", k=10)
print(f"  documents in store: {len(docs)}")
for d in docs[:3]:
    src = (d.metadata or {}).get("source", "?")
    print(f"    - {src}: {d.page_content[:60]}...")

assert docs, "knowledge base is empty - path regression!"
print()
print("PASS: store resolves to the populated index")