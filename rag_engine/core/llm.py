from rag_engine.core.config import settings


def generate_response(prompt: str, temperature: float):
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    config = genai.types.GenerationConfig(temperature=temperature)
    response = model.generate_content(prompt, generation_config=config)
    return response.text

