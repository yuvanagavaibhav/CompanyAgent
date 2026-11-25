import os
from dotenv import load_dotenv

load_dotenv()

def extract_text(content) -> str:
    """Helper to extract pure string text from LangChain message content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)
    return str(content)

def get_llm(temperature: float = 0.0):
    """
    Factory function to return the configured LLM.
    Uses Google Gemini (gemini-3.5-flash-lite) if GOOGLE_API_KEY is present,
    otherwise falls back to Groq.
    """
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # gemini-3.5-flash-lite is the active fast model
        model_name = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=temperature
        )

    elif groq_key:
        from langchain_groq import ChatGroq
        from src.config import GROQ_MODEL
        return ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=groq_key,
            temperature=temperature
        )

    else:
        raise ValueError("No API key found. Please provide GOOGLE_API_KEY or GROQ_API_KEY in your .env file.")
