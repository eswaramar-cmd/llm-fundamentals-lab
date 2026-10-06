"""Quick load test — no external dependencies beyond the standard library.

Tests the API with N concurrent users and reports p50, p95, p99,
throughput, and error rate.

Usage:
    # Start 3 API replicas first:
    docker compose -f backend/docker-compose.yml up --scale api=3 --build

    # Then run the load test:
    python load_tests/quick_test.py --users 10 --requests 50
    python load_tests/quick_test.py --users 25 --requests 100
    python load_tests/quick_test.py --users 50 --requests 200
"""

from __future__ import annotations

import argparse
import json
import ssl
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import uuid


@dataclass
class TestResult:
    """Result of a single request."""

    status_code: int
    duration: float
    user_id: str
    error: str = ""


@dataclass
class Summary:
    """Aggregated load-test results."""

    total: int = 0
    success: int = 0
    errors: int = 0
    rate_limited: int = 0
    durations: list[float] = field(default_factory=list)
    error_types: dict[str, int] = field(default_factory=dict)


def _make_request(url: str, payload: dict, timeout: int = 60) -> TestResult:
    """Send a single POST request and return the result."""

    user_id = f"lt_user_{uuid.uuid4().hex[:8]}"
    session_id = f"lt_session_{uuid.uuid4().hex[:8]}"
    payload = dict(payload)
    payload["user_id"] = user_id
    payload["session_id"] = session_id

    data = json.dumps(payload).encode("utf-8")
    start = time.monotonic()

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-User-ID": user_id,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = resp.read()
            duration = time.monotonic() - start
            return TestResult(
                status_code=resp.status,
                duration=duration,
                user_id=user_id,
            )
    except urllib.error.HTTPError as e:
        duration = time.monotonic() - start
        return TestResult(
            status_code=e.code,
            duration=duration,
            user_id=user_id,
            error=f"HTTP {e.code}",
        )
    except Exception as e:
        duration = time.monotonic() - start
        return TestResult(
            status_code=0,
            duration=duration,
            user_id=user_id,
            error=type(e).__name__,
        )


def run_load_test(url: str, num_users: int, requests_per_user: int, max_workers: int) -> Summary:
    """Run a concurrent load test."""

    summary = Summary()
    total_requests = num_users * requests_per_user

    print(f"\n{'='*60}")
    print(f"Load Test: {num_users} users × {requests_per_user} requests = {total_requests}")
    print(f"URL: {url}")
    print(f"Concurrency: {max_workers}")
    print(f"{'='*60}")

    start_all = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(total_requests):
            future = executor.submit(
                _make_request,
                url,
                {"question": "What is 2+2?", "tools": []},
            )
            futures.append(future)

        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result(timeout=120)
                summary.total += 1
                summary.durations.append(result.duration)

                if result.status_code == 200:
                    summary.success += 1
                elif result.status_code == 429:
                    summary.rate_limited += 1
                else:
                    summary.errors += 1
                    etype = result.error or f"http_{result.status_code}"
                    summary.error_types[etype] = summary.error_types.get(etype, 0) + 1

                completed += 1
                if completed % 25 == 0 or completed == total_requests:
                    elapsed = time.monotonic() - start_all
                    rps = completed / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {completed}/{total_requests} ({rps:.1f} req/s)")

            except Exception as e:
                summary.total += 1
                summary.errors += 1
                summary.error_types[type(e).__name__] = summary.error_types.get(type(e).__name__, 0) + 1

    elapsed = time.monotonic() - start_all
    return summary, elapsed


def print_summary(summary: Summary, elapsed: float, label: str):
    """Print a formatted summary report."""

    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")

    if not summary.durations:
        print("  No successful requests recorded.")
        return

    durations = sorted(summary.durations)

    def pct(p):
        if not durations:
            return 0.0
        idx = min(len(durations) - 1, int(len(durations) * p / 100))
        return durations[idx]

    rps = summary.total / elapsed if elapsed > 0 else 0
    success_rate = (summary.success / summary.total * 100) if summary.total > 0 else 0

    print(f"  Total requests : {summary.total}")
    print(f"  Successful     : {summary.success}")
    print(f"  Errors         : {summary.errors}")
    print(f"  Rate-limited   : {summary.rate_limited}")
    print(f"  Duration       : {elapsed:.1f}s")
    print(f"  Throughput     : {rps:.1f} req/s")
    print(f"  Success rate   : {success_rate:.1f}%")
    print(f"  p50 latency    : {pct(50):.2f}s")
    print(f"  p95 latency    : {pct(95):.2f}s")
    print(f"  p99 latency    : {pct(99):.2f}s")
    print(f"  Max latency    : {durations[-1]:.2f}s")
    print(f"  Min latency    : {durations[0]:.2f}s")

    if summary.error_types:
        print(f"  Error breakdown:")
        for etype, count in sorted(summary.error_types.items()):
            print(f"    {etype}: {count}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Quick load test for AI Agent API")
    parser.add_argument("--url", default="http://localhost/agent/run", help="API endpoint URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=20, help="Total requests per user")
    parser.add_argument("--workers", type=int, default=20, help="Max concurrent workers")
    parser.add_argument("--all", action="store_true", help="Run all test levels (10, 25, 50)")
    args = parser.parse_args()

    if args.all:
        for n_users, reqs in [(10, 30), (25, 60), (50, 100)]:
            summary, elapsed = run_load_test(args.url, n_users, reqs, min(n_users * 2, 50))
            print_summary(summary, elapsed, f"{n_users} concurrent users")
    else:
        summary, elapsed = run_load_test(args.url, args.users, args.requests, args.workers)
        print_summary(summary, elapsed, f"{args.users} concurrent users")


if __name__ == "__main__":
    main()
