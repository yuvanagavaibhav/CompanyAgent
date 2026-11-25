import re
from typing import Tuple, Dict, Any

# Common adversarial prompt injection and jailbreak signatures
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"(?i)system\s*prompt",
    r"(?i)reveal\s+(your\s+)?(instructions|prompt)",
    r"(?i)you\s+are\s+now\s+(unrestricted|DAN|jailbroken)",
    r"(?i)act\s+as\s+(an?\s+)?unrestricted",
    r"(?i)bypass\s+(your\s+)?safety",
    r"(?i)print\s+(the\s+)?developer\s+mode",
]

# Basic PII detection regex patterns
PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "API Key / Secret Token": r"(?i)(api[_-]?key|bearer\s+[a-z0-9_-]{20,}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{20,})",
}

class InputGuardrailException(Exception):
    pass

class OutputGuardrailException(Exception):
    pass

def validate_input(user_query: str) -> Tuple[bool, str]:
    """
    Validates user query before running RAG pipeline:
    1. Checks for empty or excessively long inputs.
    2. Detects prompt injection / jailbreak attempts.
    3. Detects exposed sensitive PII.
    
    Returns:
        (is_valid, reason_or_sanitized_query)
    """
    cleaned_query = user_query.strip()
    
    # 1. Length checks
    if not cleaned_query:
        return False, "Query cannot be empty."
    if len(cleaned_query) > 1500:
        return False, "Query is too long (maximum 1500 characters allowed)."
        
    # 2. Prompt Injection detection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned_query):
            return False, "Security Alert: Prompt injection or system prompt extraction attempt detected."
            
    # 3. Sensitive PII detection
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, cleaned_query):
            return False, f"Privacy Alert: Sensitive information ({pii_type}) detected in input. Please remove it before asking."
            
    return True, cleaned_query

def validate_output(response_text: str, context_text: str = "") -> Tuple[bool, str]:
    """
    Validates LLM-generated output:
    1. Ensures internal prompt instructions or raw API keys are not leaked.
    2. Flags empty responses.
    
    Returns:
        (is_valid, sanitized_response)
    """
    if not response_text or not response_text.strip():
        return False, "I apologize, but I could not formulate an answer based on the provided documents."

    # Check for leaked API key patterns in output
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, response_text):
            return False, "Response blocked: Potential sensitive key/credential leakage detected in output."

    return True, response_text
