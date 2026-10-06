"""Verify llm_fast resolves providers and keys from backend/.env."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from backend.agent.llm_fast import (  # noqa: E402
    available_specs,
    get_fallback_spec,
    get_primary_spec,
    provider_order,
)

print(f"  env files searched: backend/.env exists = {(ROOT / 'backend' / '.env').exists()}")
print(f"  order    : {provider_order()}")
print(f"  primary  : {get_primary_spec().public()}  key={bool(get_primary_spec().api_key)}")
print(f"  fallback : {get_fallback_spec().public()}  key={bool(get_fallback_spec().api_key)}")
print(f"  active   : {[s.name for s in available_specs()]}")

assert available_specs(), "no provider resolved - keys not found"
print()
print("PASS: llm_fast resolves a live provider")