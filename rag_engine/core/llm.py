"""Dispatcher LLM: chọn Gemini Cloud hoặc Ollama local theo cấu hình."""

from rag_engine.core.config import settings


def _generate_gemini(prompt: str, temperature: float) -> str:
    """Gọi Google Gemini API và trả về text sinh ra."""
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    config = genai.types.GenerationConfig(temperature=temperature)
    response = model.generate_content(prompt, generation_config=config)
    return response.text


def _generate_ollama(prompt: str, temperature: float) -> str:
    """Gọi Ollama server local và trả về text sinh ra."""
    import requests

    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate_response(prompt: str, temperature: float) -> str:
    """Sinh câu trả lời từ LLM đang cấu hình (ollama hoặc gemini)."""
    provider = (settings.llm_provider or "ollama").lower()
    if provider == "gemini":
        return _generate_gemini(prompt, temperature)
    return _generate_ollama(prompt, temperature)


def stream_response(prompt: str, temperature: float):
    """Trả về generator yield từng chunk text từ LLM đang cấu hình."""
    provider = (settings.llm_provider or "ollama").lower()
    if provider == "gemini":
        yield from _stream_gemini(prompt, temperature)
        return
    yield from _stream_ollama(prompt, temperature)


def _stream_ollama(prompt: str, temperature: float):
    """Stream từng chunk từ Ollama /api/generate (stream=True, JSONL)."""
    import json

    import requests

    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
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


def _stream_gemini(prompt: str, temperature: float):
    """Stream từng chunk từ Gemini API."""
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    config = genai.types.GenerationConfig(temperature=temperature)
    for chunk in model.generate_content(prompt, generation_config=config, stream=True):
        text = getattr(chunk, "text", "") or ""
        if text:
            yield text
