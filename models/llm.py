"""Multi-provider LLM integration layer."""

from __future__ import annotations
from typing import Any
import google.generativeai as genai

try:
    from groq import Groq  
except Exception:  
    Groq = None  

try:
    from openai import OpenAI
except Exception:  
    OpenAI = None  

from config.config import get_settings, validate_api_key
from utils.response_modes import build_concise_prompt, build_detailed_prompt


_ACTIVE_PROVIDER: str | None = None
_ACTIVE_CLIENT: Any | None = None


def _normalize_gemini_model_name(model_name: str) -> str:
    """Gemini SDK accepts bare model names; strip optional models/ prefix."""

    return model_name.removeprefix("models/").strip()


def _resolve_gemini_model_name(preferred_model_name: str) -> str:
    """Return a generateContent-capable Gemini model name available to this API key."""

    preferred = _normalize_gemini_model_name(preferred_model_name)
    candidate_order = [
        preferred,
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ]

    list_models_fn = getattr(genai, "list_models", None)
    if list_models_fn is None:
        return preferred

    try:
        models = list(list_models_fn())
    except Exception:
        return preferred

    supported_names: set[str] = set()
    for model in models:
        methods = getattr(model, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            model_name = _normalize_gemini_model_name(getattr(model, "name", ""))
            if model_name:
                supported_names.add(model_name)

    for candidate in candidate_order:
        if candidate in supported_names:
            return candidate

    if supported_names:
        return sorted(supported_names)[0]

    return preferred


def load_model(provider_name: str) -> Any:
    """Load the selected provider client and store it as active."""

    global _ACTIVE_PROVIDER
    global _ACTIVE_CLIENT

    provider = provider_name.strip().lower()
    settings = get_settings()

    validate_api_key(provider)

    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run: pip install openai")
        _ACTIVE_CLIENT = OpenAI(api_key=settings.openai_api_key)
    elif provider == "groq":
        if Groq is None:
            raise RuntimeError("groq package is not installed. Run: pip install groq")
        _ACTIVE_CLIENT = Groq(api_key=settings.groq_api_key)
    elif provider == "gemini":
        configure_fn = getattr(genai, "configure", None)
        if configure_fn is None:
            raise RuntimeError("google-generativeai package is not available or outdated.")
        configure_fn(api_key=settings.gemini_api_key)

        resolved_model = _resolve_gemini_model_name(settings.gemini_model_name)
        model_cls = getattr(genai, "GenerativeModel", None)
        if model_cls is None:
            raise RuntimeError("google-generativeai package is not available or outdated.")

        _ACTIVE_CLIENT = model_cls(_normalize_gemini_model_name(resolved_model))
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    _ACTIVE_PROVIDER = provider
    return _ACTIVE_CLIENT


def _build_prompt(query: str, context: str, mode: str) -> str:
    if mode.strip().lower() == "detailed":
        return build_detailed_prompt(query=query, context=context)
    return build_concise_prompt(query=query, context=context)


def _get_active_model_name() -> str:
    settings = get_settings()
    model_by_provider = {
        "openai": settings.openai_model_name,
        "groq": settings.groq_model_name,
        "gemini": settings.gemini_model_name,
    }
    if _ACTIVE_PROVIDER not in model_by_provider:
        raise RuntimeError(f"Unknown active provider: {_ACTIVE_PROVIDER}")
    return model_by_provider[_ACTIVE_PROVIDER]


def _generate_openai_like_response(prompt: str) -> str:
    """Handle providers that use OpenAI-compatible chat completions."""

    model_name = _get_active_model_name()
    client = _ACTIVE_CLIENT
    if client is None:
        raise RuntimeError("No active client available.")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    content = completion.choices[0].message.content
    return content.strip() if content else "I could not generate a response."


def generate_response(query: str, context: str, mode: str) -> str:
    """Generate a response using the currently loaded provider."""

    if _ACTIVE_PROVIDER is None or _ACTIVE_CLIENT is None:
        raise RuntimeError("No active model loaded. Call load_model(provider_name) first.")

    prompt = _build_prompt(query=query, context=context, mode=mode)

    try:
        if _ACTIVE_PROVIDER in {"openai", "groq"}:
            return _generate_openai_like_response(prompt)

        if _ACTIVE_PROVIDER == "gemini":
            response = _ACTIVE_CLIENT.generate_content(prompt)
            text = getattr(response, "text", None)
            return text.strip() if text else "I could not generate a response."

        raise RuntimeError(f"No handler implemented for provider: {_ACTIVE_PROVIDER}")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate response: {exc}") from exc
