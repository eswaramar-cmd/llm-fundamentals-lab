"""Timestamps every SSE frame from /chat/stream to locate latency.

Usage:
    venv\\Scripts\\python.exe profile_stream.py "your question"

Prints a per-frame timeline plus a per-node summary, so optimisation targets the
node that actually costs time instead of the one that looks suspicious.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8010/chat/stream"


def run(question: str, url: str = URL) -> None:
    payload = json.dumps(
        {"message": question, "user_id": "profiler", "thread_id": "profile"}
    ).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    started = time.perf_counter()
    node_started: dict[str, float] = {}
    rows: list[tuple[float, str, str]] = []
    first_token_at: float | None = None
    done_payload: dict | None = None

    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            kind = event.get("type", "?")

            if kind == "status":
                node = event.get("node", "?")
                if node in node_started:
                    rows.append(
                        (node_started.pop(node), "node_end", f"{node} ({event.get('message')})")
                    )
                node_started[node] = elapsed_ms
                rows.append((elapsed_ms, "status", f"{node} ({event.get('message')})"))
            elif kind == "token":
                if first_token_at is None:
                    first_token_at = elapsed_ms
                rows.append((elapsed_ms, "token", repr(event.get("text", ""))[:56]))
            elif kind == "tool_start":
                rows.append((elapsed_ms, "tool_start", str(event.get("tool"))))
            elif kind == "tool_end":
                rows.append((elapsed_ms, "tool_end", str(event.get("tool"))))
            elif kind == "done":
                done_payload = event
                rows.append((elapsed_ms, "done", json.dumps(event)[:90]))
            elif kind == "error":
                rows.append((elapsed_ms, "error", str(event.get("message"))))

    total_ms = (time.perf_counter() - started) * 1000

    print(f"\nquestion: {question!r}")
    print("=" * 78)
    for at, kind, detail in rows:
        marker = "  " if kind == "token" else "* "
        print(f"{marker}{at:8.0f}ms  {kind:<11} {detail}")

    print("-" * 78)
    if first_token_at is not None:
        print(f"time to first token : {first_token_at:.0f}ms")
    print(f"total wall time     : {total_ms:.0f}ms")

    if done_payload:
        print(f"reported latency_ms : {done_payload.get('latency_ms')}")
        print(f"ttft_ms             : {done_payload.get('time_to_first_token_ms')}")
        print(f"chunks              : {done_payload.get('chunks')}")
        print(f"chars               : {done_payload.get('chars')}")
        print(f"chars_per_second    : {done_payload.get('chars_per_second')}")
        print(f"model               : {done_payload.get('model')}")
        print(f"answer chars        : {len(done_payload.get('answer', ''))}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Name one LangGraph concept."
    port = sys.argv[1] if sys.argv[0].endswith("profile_stream.py") and False else 8010
    run(question, f"http://127.0.0.1:{port}/chat/stream")