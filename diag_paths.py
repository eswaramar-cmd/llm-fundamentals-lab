"""Verify config paths resolve correctly after moving agent/ under backend/."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from backend.agent.config import (  # noqa: E402
    BACKEND_DIR,
    ENV_FILES,
    PROJECT_ROOT,
    get_settings,
)

print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
print(f"  BACKEND_DIR  : {BACKEND_DIR}")
print(f"  env files    : {[str(p) for p in ENV_FILES]}")
print()

settings = get_settings()

checks = {
    "chroma_rag exists at resolved path": Path(settings.chroma_rag_path).exists(),
    "documents dir exists": Path(settings.documents_path).exists(),
    "chroma_rag is under PROJECT_ROOT": str(settings.chroma_rag_path).startswith(str(PROJECT_ROOT)),
    "not under backend/": "chroma_rag" not in settings.chroma_rag_path.replace(str(BACKEND_DIR), ""),
}

for k, v in checks.items():
    print(f"  {'ok ' if v else 'FAIL'} {k}")

print()
print(f"  chroma_rag path   : {settings.chroma_rag_path}")
print(f"  chroma_memory     : {settings.chroma_memory_path}")
print(f"  documents         : {settings.documents_path}")
print(f"  LLM_PROVIDER      : {settings.llm_provider}")
print(f"  provider order    : {os.environ.get('LLM_PROVIDER_ORDER', '(not in env)')}")

failures = [k for k, v in checks.items() if not v]
print()
print("RESULT:", "PASS" if not failures else f"FAIL -> {failures}")
raise SystemExit(1 if failures else 0)