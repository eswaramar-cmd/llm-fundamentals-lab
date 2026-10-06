"""LLM provider factory — supports Ollama, OpenAI, Anthropic, Groq.

Centralised so every node uses the same LLM instance. The provider
is selected via ``LLM_PROVIDER``:

    Development (default):
        LLM_PROVIDER=ollama
        LLM_BASE_URL=http://host.docker.internal:11434
        LLM_MODEL=llama3.2:3b

    Production (example):
        LLM_PROVIDER=openai
        LLM_MODEL=gpt-4o
        LLM_BASE_URL=https://api.openai.com/v1  (optional)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from backend.agent.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached chat model (no tools bound).

    A configured cloud provider wins over local Ollama: when ``GROQ_API_KEY`` is
    set, ``llama-3.1-8b-instant`` is used, and ``GOOGLE_API_KEY`` alone selects
    Gemini. Local Ollama stays the default so the app still runs with no keys.
    """

    from backend.agent.llm_fast import available_specs, build_client

    cloud = available_specs()

    if cloud:
        try:
            return build_client(cloud[0])
        except RuntimeError as exc:
            logger.warning(
                "Cloud provider %s unavailable (%s); using configured provider",
                cloud[0].name,
                exc,
            )

    settings = get_settings()
    provider = (settings.llm_provider or "ollama").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_num_predict,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_num_predict,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_num_predict,
        )

    # Default: Ollama
    from langchain_ollama import ChatOllama

    base_url = settings.llm_base_url or settings.ollama_base_url
    logger.info("Using Ollama LLM: %s at %s", settings.llm_model, base_url)
    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.ollama_temperature,
        base_url=base_url,
        num_predict=settings.ollama_num_predict,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )


@lru_cache(maxsize=1)
def get_llm_capped() -> BaseChatModel:
    """Chat model for parallel lanes: tighter budget, optional smaller model.

    A cloud provider is already fast and is trusted to handle the request, so it
    is reused as-is. Only local Ollama gets the downshift, because a small model
    cannot reliably emit structured tool calls.
    """

    settings = get_settings()
    provider = (settings.llm_provider or "ollama").lower()

    if provider in {"openai", "anthropic", "groq"}:
        return get_llm()

    from langchain_ollama import ChatOllama

    base_url = settings.llm_base_url or settings.ollama_base_url
    model = settings.parallel_llm_model or settings.llm_model

    logger.info("Using lane LLM: %s at %s", model, base_url)

    return ChatOllama(
        model=model,
        temperature=settings.ollama_temperature,
        base_url=base_url,
        num_predict=settings.parallel_num_predict,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )


@lru_cache(maxsize=1)
def get_llm_with_tools() -> BaseChatModel:
    """Return a cached chat model with all tools bound."""
    from backend.agent.tools import DEFAULT_TOOLS

    return get_llm().bind_tools(DEFAULT_TOOLS)


# Which tools each classified intent can actually reach. Binding all four costs
# ~520 tokens of schema on every call, and on a CPU-only box prompt evaluation
# is the dominant cost, so a narrow slice is a large latency win.
_INTENT_TOOLS: dict[str, tuple[str, ...]] = {
    "calculate": ("calculator",),
    "rag": ("knowledge_base_search",),
    "research": ("web_search",),
    # `direct` means the router already decided no tool is needed, and the
    # router has a dedicated `calculate` intent for arithmetic. Binding the
    # calculator here shipped a tool schema on no-tool questions and pushed
    # Gemini onto its automatic-function-calling path, which is markedly
    # slower. `_bind_tools_cached(())` returns the bare model.
    "direct": (),
    "sensitive": (),
    # The only intent that can send mail. Bound alone so an unrelated question
    # cannot trigger a real send.
    "email": ("send_email",),
    # Incident investigation intent binds the investigate_incident tool
    "incident": ("investigate_incident",),
}


@lru_cache(maxsize=8)
def _bind_tools_cached(tool_names: tuple[str, ...]):
    """Bind a specific tool subset. Cached because binding is not free."""
    from backend.agent.tools import ALL_TOOLS

    selected = [ALL_TOOLS[n] for n in tool_names if n in ALL_TOOLS]

    if not selected:
        return get_llm()

    return get_llm().bind_tools(selected)


def get_llm_for_intent(intent: str) -> BaseChatModel:
    """Return a model bound only to the tools this intent can use.

    Falls back to the full tool set for any unrecognised intent, so a new
    intent never silently loses the ability to call a tool.
    """

    if not intent:
        return get_llm_with_tools()

    tool_names = _INTENT_TOOLS.get(intent)

    if tool_names is None:
        return get_llm_with_tools()

    if not tool_names:
        return get_llm()

    return _bind_tools_cached(tool_names)


@lru_cache(maxsize=1)
def get_local_llm() -> BaseChatModel:
    """The configured local Ollama model, ignoring any cloud key.

    Used as the last-resort provider. It is the only option that cannot return
    503, and on a CPU-only box it answers short factual prompts faster than a
    saturated cloud endpoint.
    """

    settings = get_settings()
    base_url = settings.llm_base_url or settings.ollama_base_url

    logger.info("Local fallback LLM: %s at %s", settings.llm_model, base_url)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.ollama_temperature,
        base_url=base_url,
        num_predict=settings.ollama_num_predict,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )


def _bind_to(model: BaseChatModel, tool_names: tuple[str, ...]) -> BaseChatModel:
    """Bind a tool subset to a specific model instance."""

    if not tool_names:
        return model

    from backend.agent.tools import ALL_TOOLS

    selected = [ALL_TOOLS[n] for n in tool_names if n in ALL_TOOLS]

    if not selected:
        return model

    return model.bind_tools(selected)


def models_for_intent(intent: str) -> list[tuple[str, BaseChatModel]]:
    """Every usable model for this intent, in fallback order.

    Ordered fastest-and-most-likely-to-succeed first: configured cloud
    providers, then local Ollama last.

    The agent node walks this list on retry. A cloud provider that returns 503
    (which the configured Google key currently does under load) then fails over
    to the next provider instead of retrying the same throttled endpoint, which
    is what made answers unreliable.
    """

    # Imported here rather than at module scope because the tools package pulls
    # in the vector store, which reads settings on import.
    from backend.agent.tools import DEFAULT_TOOLS

    all_tools = tuple(tool.name for tool in DEFAULT_TOOLS)

    if not intent:
        tool_names: tuple[str, ...] = all_tools
    else:
        tool_names = _INTENT_TOOLS.get(intent, all_tools)

    candidates: list[tuple[str, BaseChatModel]] = []

    try:
        from backend.agent.llm_fast import available_specs, build_client

        for spec in available_specs():
            try:
                candidates.append((spec.name, build_client(spec)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Provider %s unavailable: %s", spec.name, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not enumerate cloud providers: %s", exc)

    if not candidates:
        # No cloud key configured, so local is the primary rather than a
        # fallback. Avoid listing it twice.
        try:
            return [("ollama", _bind_to(get_llm(), tool_names))]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Primary local model unavailable: %s", exc)

    return [(name, _bind_to(model, tool_names)) for name, model in candidates]

