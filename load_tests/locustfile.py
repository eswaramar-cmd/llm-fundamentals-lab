"""Load tests for the AI Research & Knowledge Agent.

Run with Locust:

    # Terminal 1 — start the agent (Docker)
    docker compose -f backend/docker-compose.yml up --scale api=3 --build

    # Terminal 2 — run load test (10 concurrent users)
    locust -f load_tests/locustfile.py --headless \
        -u 10 -r 2 --host http://localhost \
        --run-time 2m --csv load_tests/results_10

    # 25 concurrent users
    locust -f load_tests/locustfile.py --headless \
        -u 25 -r 5 --host http://localhost \
        --run-time 2m --csv load_tests/results_25

    # 50 concurrent users
    locust -f load_tests/locustfile.py --headless \
        -u 50 -r 10 --host http://localhost \
        --run-time 2m --csv load_tests/results_50

    # Interactive Web UI (default: http://localhost:8089)
    locust -f load_tests/locustfile.py --host http://localhost

Requirements:
    pip install locust
"""

from __future__ import annotations

import json
import os
import random
import time
from locust import HttpUser, task, between, events


# ------------------------------------------------------------------
# Unique user IDs for isolation testing
# ------------------------------------------------------------------

_USER_ID_COUNTER = [0]


def _next_user_id() -> str:
    _USER_ID_COUNTER[0] += 1
    return f"loadtest_user_{_USER_ID_COUNTER[0]}"


# ------------------------------------------------------------------
# Test questions — varied to exercise different code paths
# ------------------------------------------------------------------

CALCULATOR_QUESTIONS = [
    "What is 25 * 40 + 100?",
    "Calculate 1567 / 34",
    "What is 2^10?",
    "How many seconds in a day?",
    "What is 17 * 23?",
]

DIRECT_QUESTIONS = [
    "Hello, what is your name?",
    "What is the capital of France?",
    "Explain what machine learning is.",
    "What are the benefits of renewable energy?",
    "Tell me about the importance of sleep.",
]

ALL_QUESTIONS = CALCULATOR_QUESTIONS + DIRECT_QUESTIONS


# ------------------------------------------------------------------
# Metrics collection for post-run summary
# ------------------------------------------------------------------

_request_durations: dict[str, list[float]] = {}
_error_counts: dict[str, int] = {}
_rate_limited = 0


class AgentUser(HttpUser):
    """Simulate a multi-user chat session with unique isolation."""

    wait_time = between(1, 3)
    user_id: str = ""
    session_id: str = ""

    def on_start(self):
        """Initialise per-user session with unique user_id."""
        self.user_id = _next_user_id()
        self.session_id = f"session_{self.user_id}"

    @task
    def query_agent(self):
        """POST /agent/run with a random question."""

        question = random.choice(ALL_QUESTIONS)

        start = time.monotonic()
        with self.client.post(
            "/agent/run",
            json={
                "question": question,
                "user_id": self.user_id,
                "session_id": self.session_id,
            },
            name="/agent/run",
            catch_exceptions=True,
        ) as resp:
            duration = time.monotonic() - start
            _request_durations.setdefault("/agent/run", []).append(duration)

            if resp.status_code == 200:
                data = resp.json()
                # Verify user isolation — response should reference our user
                assert data.get("session_id") == self.session_id, (
                    f"Session mismatch: expected {self.session_id}, got {data.get('session_id')}"
                )
            elif resp.status_code == 429:
                global _rate_limited
                _rate_limited += 1
            else:
                _error_counts[f"http_{resp.status_code}"] = (
                    _error_counts.get(f"http_{resp.status_code}", 0) + 1
                )

    @task
    def health_check(self):
        """GET /health — lightweight liveness probe."""
        with self.client.get("/health", name="/health", catch_exceptions=True) as resp:
            if resp.status_code != 200:
                _error_counts["health_failed"] = _error_counts.get("health_failed", 0) + 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print a summary report when the test finishes."""
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)

    total_requests = sum(len(v) for v in _request_durations.values())
    total_errors = sum(_error_counts.values()) + _rate_limited

    for endpoint, durations in sorted(_request_durations.items()):
        if durations:
            p50 = _percentile(durations, 50)
            p95 = _percentile(durations, 95)
            p99 = _percentile(durations, 99)
            avg = sum(durations) / len(durations)
            rps = len(durations) / (sum(durations) / len(durations)) if durations else 0
            errors = _error_counts.get(f"http_500", 0) + _error_counts.get(f"http_502", 0)
            print(f"\n  {endpoint}")
            print(f"    Total requests : {len(durations)}")
            print(f"    p50 latency    : {p50:.2f}s")
            print(f"    p95 latency    : {p95:.2f}s")
            print(f"    p99 latency    : {p99:.2f}s")
            print(f"    avg latency    : {avg:.2f}s")
            print(f"    errors         : {errors}")
            print(f"    rate-limited   : {_rate_limited}")

    for err_type, count in sorted(_error_counts.items()):
        print(f"  Error {err_type}: {count}")

    if total_requests > 0:
        success_rate = ((total_requests - total_errors) / total_requests) * 100
        print(f"\n  Overall success rate: {success_rate:.1f}%")

    print("=" * 60)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]
