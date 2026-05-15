import google.generativeai as genai
from app.core.config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

def generate_response(prompt, temperature):
    model = genai.GenerativeModel('gemini-2.5-flash')

    config = genai.types.GenerationConfig(
        temperature=temperature,
    )
    
    response = model.generate_content(prompt, generation_config=config)

    return response.text