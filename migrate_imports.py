"""Rewrite absolute `agent.*` imports to `backend.agent.*` after the move.

Every module used an absolute `from agent.x import y`, so moving the package
under backend/ breaks them all. This rewrites import statements, bare `import
agent.x`, and patch/monkeypatch target strings such as "agent.llm.ChatOllama",
which are string literals that no import fixer would catch.
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

TARGETS = [
    ROOT / "backend",
    ROOT / "tests",
    ROOT / "load_tests",
]

SKIP_DIRS = {"__pycache__", "venv", "node_modules", ".pytest_cache", "chroma_rag", "chroma_memory"}

# Ordered so longer, more specific patterns are handled first.
PATTERNS = [
    # from agent.sub.module import X  ->  from backend.agent.sub.module import X
    (re.compile(r"^(\s*)from\s+agent(\.[A-Za-z0-9_.]+)?\s+import\s+", re.M), r"\1from backend.agent\2 import "),
    # import agent.sub.module
    (re.compile(r"^(\s*)import\s+agent(\.[A-Za-z0-9_.]+)?\s*$", re.M), r"\1import backend.agent\2"),
    # import agent.sub.module, other  (comma form)
    (re.compile(r"^(\s*)import\s+agent(\.[A-Za-z0-9_.]+)?\s*,", re.M), r"\1import backend.agent\2,"),
    # patch()/monkeypatch string targets: "agent.llm.ChatOllama"
    (re.compile(r'("|\')agent(\.[A-Za-z0-9_.]+)+(\1)'), r'"backend.agent\2"'),
]

changed = []
total = 0

for target in TARGETS:
    if not target.exists():
        continue

    for path in sorted(target.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        original = path.read_text(encoding="utf-8")
        text = original

        for pattern, repl in PATTERNS:
            text = pattern.sub(repl, text)

        if text != original:
            n = sum(1 for a, b in zip(original.splitlines(), text.splitlines()) if a != b)
            path.write_text(text, encoding="utf-8")
            changed.append((path.relative_to(ROOT), n))
            total += n

print(f"files rewritten : {len(changed)}")
print(f"lines changed   : {total}")
print()
for rel, n in changed:
    print(f"  {str(rel):58} {n:>3} lines")

print()
leftover = []
for target in TARGETS:
    if not target.exists():
        continue
    for path in target.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(^|\W)agent\.", line) and "backend.agent" not in line:
                leftover.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")

print(f"leftover bare 'agent.' references: {len(leftover)}")
for line in leftover[:25]:
    print("  ", line)