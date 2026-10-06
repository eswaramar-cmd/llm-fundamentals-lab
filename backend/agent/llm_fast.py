"""Fast LLM providers for low-latency streaming.

Two free cloud providers, tried in the order given by ``LLM_PROVIDER_ORDER``:

1. **Groq** ``llama-3.1-8b-instant`` -- primary. Purpose-built LPU inference,
   typically first token in well under a second.
2. **Google Gemini** ``gemini-3.1-flash-lite`` -- fallback when Groq has no key,
   errors, or times out.

   Note: the original spec asked for ``gemini-1.5-flash``. Google has retired
   it and it now answers ``404 model not found``; verified against the live
   ModelService for this key, which offers gemini-2.5-flash and newer.
   ``gemini-3.1-flash-lite`` was measured fastest of the available models at
   roughly 1s to first token, versus roughly 13s for gemini-2.5-flash.

Both are accessed through LangChain's async interfaces so tokens can be
streamed to the client rather than buffered.

Nothing here is imported at module load: the provider SDKs are optional, and a
missing key must degrade to a clear runtime error rather than an import crash
that takes the whole app down.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

# Load the same files pydantic reads, so a direct import of this module (no
# Settings involved) still sees the keys. find_dotenv walks upward, which
# covers both the repository root and backend/.
load_dotenv(find_dotenv(usecwd=False))
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

# Provider order is explicit. Set LLM_PROVIDER_ORDER=groq,google to prefer
# Groq, or a single value to pin one provider.
DEFAULT_PROVIDER_ORDER = "groq, google"

DEFAULT_PRIMARY_MODEL = "llama-3.1-8b-instant"

# gemini-1.5-flash from the original spec is retired by Google and returns
# "404 model not found". gemini-3.1-flash-lite is the fastest model that
# still answers, at roughly 1s to first token.
DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-lite"

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1000

# A stalled provider should not hold a request open indefinitely.
CALL_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int

    def public(self) -> dict[str, Any]:
        """Safe-to-serialise view. Never includes the key."""
        return {"provider": self.name, "model": self.model}


def _key(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _num(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def get_primary_spec() -> ProviderSpec:
    model = os.getenv("GROQ_MODEL", "").strip()
    # Auto-sanitize invalid/deprecated model names to high-speed LPU model
    if not model or model.startswith("openai/") or "gpt" in model.lower():
        model = "llama-3.3-70b-versatile"

    return ProviderSpec(
        name="groq",
        model=model,
        api_key=_key("GROQ_API_KEY"),
        temperature=_float("GROQ_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=_num("GROQ_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    )


def get_fallback_spec() -> ProviderSpec:
    return ProviderSpec(
        name="google",
        model=os.getenv("GOOGLE_MODEL", "").strip() or DEFAULT_FALLBACK_MODEL,
        api_key=_key("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        temperature=_float("GOOGLE_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=_num("GOOGLE_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    )


def _spec_by_name() -> dict[str, ProviderSpec]:
    return {
        "groq": get_primary_spec(),
        "google": get_fallback_spec(),
    }


def provider_order() -> list[str]:
    """Provider priority, from ``LLM_PROVIDER_ORDER``.

    Reading priority from config rather than inferring it from which keys are
    present means a stray key cannot silently change which model answers.
    """
    raw = os.getenv("LLM_PROVIDER_ORDER", "").strip() or DEFAULT_PROVIDER_ORDER

    order: list[str] = []
    for name in raw.replace(";", ",").split(","):
        name = name.strip().lower()
        if name and name not in order:
            order.append(name)

    known = {"groq", "google"}
    unknown = [n for n in order if n not in known]
    if unknown:
        logger.warning(
            "Ignoring unknown providers in LLM_PROVIDER_ORDER: %s",
            ", ".join(unknown),
        )

    # Drop unknown names, then append any provider not mentioned so a typo
    # never silently disables a configured fallback.
    result = [n for n in order if n in known]
    result.extend(sorted(known - set(result)))
    return result


def available_specs() -> list[ProviderSpec]:
    """Configured providers, in priority order, with no-key entries removed."""
    specs = _spec_by_name()
    return [specs[n] for n in provider_order() if specs[n].api_key]


# ---------------------------------------------------------
# Client construction
# ---------------------------------------------------------


def build_client(spec: ProviderSpec):
    """Return an async LangChain chat client for ``spec``.

    Raises ``RuntimeError`` with an actionable message when the SDK or key is
    missing, so the caller can report it instead of crashing at import time.
    """

    if not spec.api_key:
        raise RuntimeError(
            f"No API key for provider '{spec.name}'. "
            f"Set {'GROQ_API_KEY' if spec.name == 'groq' else 'GOOGLE_API_KEY'}."
        )

    if spec.name == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError(
                "langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            ) from exc

        return ChatGroq(
            model=spec.model,
            api_key=spec.api_key,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            streaming=True,
        )

    if spec.name == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=spec.model,
            google_api_key=spec.api_key,
            temperature=spec.temperature,
            max_output_tokens=spec.max_tokens,
        )

    raise RuntimeError(f"Unknown provider '{spec.name}'")


# ---------------------------------------------------------
# Streaming with timeout, retry and fallback
# ---------------------------------------------------------


async def astream_with_fallback(
    messages: list,
    tools: list | None = None,
    specs: list[ProviderSpec] | None = None,
    on_fallback=None,
):
    """Stream tokens from the first provider that works.

    Yields raw LangChain chunks so the caller can forward model tokens
    untouched -- no buffering, no fake typing.

    Retries ``MAX_RETRIES`` times per provider on transient failure, then moves
    to the next provider. Raises the last error only once every provider is
    exhausted, so a missing Gemini key never masks a working Groq call.
    """

    candidates = specs if specs is not None else available_specs()

    if not candidates:
        raise RuntimeError(
            "No LLM provider configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env"
        )

    last_error: Exception | None = None

    for spec in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = build_client(spec)

                if tools:
                    client = client.bind_tools(tools)

                logger.info(
                    "Streaming from %s/%s (attempt %d/%d)",
                    spec.name,
                    spec.model,
                    attempt,
                    MAX_RETRIES,
                )

                produced = False

                async def _drive():
                    nonlocal produced
                    async for chunk in client.astream(messages):
                        produced = True
                        yield chunk

                iterator = _drive()

                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=CALL_TIMEOUT_SECONDS,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        # Timed out mid-stream: fall back rather than hang.
                        await iterator.aclose()
                        raise TimeoutError(
                            f"{spec.name} stalled for {CALL_TIMEOUT_SECONDS}s"
                        )

                    yield chunk

                if produced:
                    return

                last_error = RuntimeError(f"{spec.name} returned no output")

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "%s/%s attempt %d/%d failed: %s",
                    spec.name,
                    spec.model,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

    raise RuntimeError(
        f"All LLM providers failed. Last error: {last_error}"
    ) from last_error


def _chunk_text(chunk) -> str:
    """Pull plain text out of a provider chunk.

    Gemini returns ``content`` as a list of blocks (reasoning, tool calls, and
    plain text) on newer models, and as a plain string on older ones, so both
    shapes have to be handled or the content cannot be concatenated.
    """

    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Plain text blocks carry "text". Blocks for tool calls and
                # reasoning carry other keys and are not user-visible output.
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)

        return "".join(parts)

    return ""


async def probe_latency(spec: ProviderSpec) -> dict[str, Any]:
    """Measure real round-trip and first-token latency for one provider."""

    result: dict[str, Any] = {
        **spec.public(),
        "configured": bool(spec.api_key),
        "ok": False,
        "total_ms": None,
        "first_token_ms": None,
        "error": None,
    }

    if not spec.api_key:
        result["error"] = "missing API key"
        return result

    started = time.monotonic()

    try:
        client = build_client(spec)
        first: float | None = None
        text = ""

        async for chunk in client.astream([("human", "Reply with the single word: ok")]):
            if first is None and _chunk_text(chunk):
                first = (time.monotonic() - started) * 1000
            text += _chunk_text(chunk)

        result["total_ms"] = round((time.monotonic() - started) * 1000)
        result["first_token_ms"] = round(first) if first is not None else None
        result["sample"] = text.strip()[:40]
        result["ok"] = bool(text.strip())

    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result