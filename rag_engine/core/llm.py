"""LLM dispatcher with runtime fallback between Gemini and Ollama."""

from __future__ import annotations

from rag_engine.core.config import settings


class LLMConfigurationError(RuntimeError):
    """Raised when no usable LLM provider is configured or reachable."""


def _generate_gemini(prompt: str, temperature: float) -> str:
    """Call the Gemini API and return the generated text."""
    if not settings.google_api_key:
        raise LLMConfigurationError("GOOGLE_API_KEY is not configured.")

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    config = genai.types.GenerationConfig(temperature=temperature)
    response = model.generate_content(prompt, generation_config=config)
    return response.text


def _generate_ollama(prompt: str, temperature: float) -> str:
    """Call the local Ollama server and return the generated text."""
    import requests

    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise LLMConfigurationError(_build_ollama_unavailable_message()) from exc
    return resp.json().get("response", "").strip()


def generate_response(prompt: str, temperature: float) -> str:
    """Generate a response and fall back to the other provider on failure."""
    errors: list[tuple[str, Exception]] = []
    for provider in _resolve_providers():
        try:
            return _generate_with_provider(provider, prompt, temperature)
        except Exception as exc:  # pragma: no cover - depends on provider availability
            errors.append((provider, exc))
    raise LLMConfigurationError(_build_all_providers_failed_message(errors))


def stream_response(prompt: str, temperature: float):
    """Yield streamed chunks and fall back before any token is emitted."""
    errors: list[tuple[str, Exception]] = []
    for provider in _resolve_providers():
        emitted = False
        try:
            for chunk in _stream_with_provider(provider, prompt, temperature):
                emitted = True
                yield chunk
            return
        except Exception as exc:  # pragma: no cover - depends on provider availability
            if emitted:
                raise LLMConfigurationError(
                    _build_stream_interrupted_message(provider, exc)
                ) from exc
            errors.append((provider, exc))
    raise LLMConfigurationError(_build_all_providers_failed_message(errors))


def _stream_ollama(prompt: str, temperature: float):
    """Stream chunks from Ollama /api/generate using JSONL."""
    import json

    import requests

    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except requests.RequestException as exc:
        raise LLMConfigurationError(_build_ollama_unavailable_message()) from exc


def _stream_gemini(prompt: str, temperature: float):
    """Stream chunks from the Gemini API."""
    if not settings.google_api_key:
        raise LLMConfigurationError("GOOGLE_API_KEY is not configured.")

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    config = genai.types.GenerationConfig(temperature=temperature)
    for chunk in model.generate_content(prompt, generation_config=config, stream=True):
        text = getattr(chunk, "text", "") or ""
        if text:
            yield text


def _generate_with_provider(provider: str, prompt: str, temperature: float) -> str:
    """Dispatch a non-streaming request to one provider."""
    if provider == "gemini":
        return _generate_gemini(prompt, temperature)
    return _generate_ollama(prompt, temperature)


def _stream_with_provider(provider: str, prompt: str, temperature: float):
    """Dispatch a streaming request to one provider."""
    if provider == "gemini":
        yield from _stream_gemini(prompt, temperature)
        return
    yield from _stream_ollama(prompt, temperature)


def _resolve_providers() -> list[str]:
    """Return ordered provider candidates based on config and readiness."""
    provider = (settings.llm_provider or "auto").lower()
    if provider not in {"auto", "gemini", "ollama"}:
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER='{settings.llm_provider}'. Use auto, gemini, or ollama."
        )

    preferred_order = ["gemini", "ollama"] if provider != "ollama" else ["ollama", "gemini"]
    candidates = [name for name in preferred_order if _is_provider_ready(name)]
    if candidates:
        return candidates
    raise LLMConfigurationError(_build_no_provider_message())


def _is_provider_ready(provider: str) -> bool:
    """Check whether a provider is configured well enough to try."""
    if provider == "gemini":
        return bool(settings.google_api_key)
    if provider == "ollama":
        return _is_ollama_available()
    return False


def _is_ollama_available() -> bool:
    """Check whether the configured Ollama server is reachable."""
    import requests

    url = f"{settings.ollama_host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        return False


def _build_no_provider_message() -> str:
    """Return an actionable error when no provider is ready."""
    return (
        "Chua co LLM san sang. Hay thuc hien mot trong hai cach: "
        "1) them GOOGLE_API_KEY vao file .env de dung Gemini, "
        "hoac 2) cai va chay Ollama tai "
        f"{settings.ollama_host}."
    )


def _build_ollama_unavailable_message() -> str:
    """Return an actionable error when Ollama is configured but unreachable."""
    return (
        "Khong the ket noi toi Ollama. Hay bat Ollama tai "
        f"{settings.ollama_host} hoac doi LLM_PROVIDER=gemini va cau hinh GOOGLE_API_KEY."
    )


def _build_all_providers_failed_message(errors: list[tuple[str, Exception]]) -> str:
    """Return one combined error after all configured providers have failed."""
    if not errors:
        return _build_no_provider_message()

    details = "; ".join(
        f"{provider}: {_summarize_error(exc)}" for provider, exc in errors
    )
    return f"Khong the su dung ca Gemini lan Ollama. Chi tiet: {details}"


def _build_stream_interrupted_message(provider: str, exc: Exception) -> str:
    """Return an error for streams that already emitted output before failing."""
    return (
        f"Luong phan hoi tu {provider} bi gian doan sau khi da bat dau tra loi: "
        f"{_summarize_error(exc)}"
    )


def _summarize_error(exc: Exception) -> str:
    """Convert provider exceptions to short user-facing text."""
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"
