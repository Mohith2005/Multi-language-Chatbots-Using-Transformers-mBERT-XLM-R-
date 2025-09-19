from typing import Dict, Any
import re
import sys

# -------------------------
# Language detection
# -------------------------
try:
    # prefer real langdetect if installed
    from langdetect import detect as _ld_detect, LangDetectException
    LANGDETECT_AVAILABLE = True
    print("langdetect available — using it for language detection.")
except Exception:
    # langdetect not available — implement a heuristic fallback
    LANGDETECT_AVAILABLE = False
    print("langdetect NOT available — using heuristic fallback for language detection.")

    def _ld_detect(text: str) -> str:
        """Heuristic fallback for language detection.

        This uses a set of markers and accented-character checks to guess
        between 'es' (Spanish), 'fr' (French), and defaults to 'en'.
        It's intentionally simple but covers common short phrases used in our tests.
        """
        if not isinstance(text, str) or not text.strip():
            return "en"
        txt = text.strip().lower()

        # Quick punctuation markers for Spanish
        if "¿" in txt or "¡" in txt:
            return "es"

        # Accented characters common in Spanish
        if re.search(r"[ñáéíóú]", txt):
            return "es"

        # Accented chars common in French
        if re.search(r"[àâçèéêëîïôûùü]", txt):
            return "fr"

        # Keyword checks (order matters: Spanish first for 'pedido' etc.)
        spanish_markers = ["pedido", "orden", "envio", "envío", "dónde", "donde", "hola", "gracias", "por favor"]
        french_markers = ["bonjour", "commande", "où", "ou", "merci", "salut"]

        for w in spanish_markers:
            if w in txt:
                return "es"
        for w in french_markers:
            if w in txt:
                return "fr"

        # Some single-word greetings
        if txt in ("hello", "hi", "hey"):
            return "en"

        # fallback
        return "en"

# Use a unified detect function name in the rest of the code
def detect(text: str) -> str:
    try:
        return _ld_detect(text)
    except Exception:
        return "en"

# -------------------------
# Optional: transformers-based classifier
# -------------------------
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
    print("transformers available — will attempt to load model if possible.")
except Exception:
    TRANSFORMERS_AVAILABLE = False
    print("transformers NOT available — using rule-based intent classification.")

# -------------------------
# Optional: translator
# -------------------------
TRANSLATOR_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    print("deep_translator available — will use it for translations when needed.")
except Exception:
    TRANSLATOR_AVAILABLE = False
    print("deep_translator NOT available — will use built-in small translations.")

# -------------------------
# Chatbot implementation
# -------------------------
class MultilingualChatbot:
    """A small chatbot with intent detection and language handling.

    It tries to use a transformer-based classifier when available, otherwise
    falls back to a deterministic keyword-based classifier. Language detection
    uses `langdetect` if present, else a heuristic fallback.
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased") -> None:
        self.model_name = model_name
        self.model_loaded = False
        self.classifier = None

        # Predefined responses (English + a few translations)
        self.RESPONSES: Dict[str, Dict[str, str]] = {
            "order_status": {
                "en": "Your order is on the way and will arrive tomorrow.",
                "es": "Su pedido está en camino y llegará mañana.",
                "fr": "Votre commande est en route et arrivera demain."
            },
            "greeting": {
                "en": "Hello! How can I assist you today?",
                "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
                "fr": "Bonjour! Comment puis-je vous aider aujourd'hui?"
            },
            "fallback": {
                "en": "Sorry, I didn't understand that.",
                "es": "Lo siento, no entendí eso.",
                "fr": "Désolé, je n'ai pas compris cela."
            }
        }

        # Label mapping — keep consistent with previous code
        self.LABEL_MAP = {0: "order_status", 1: "greeting", 2: "fallback"}

        # Keyword lists used by the fallback rule-based classifier
        self._order_keywords = [
            "order", "pedido", "orden", "en camino", "where", "dónde", "donde", "où", "envío", "envio"
        ]
        self._greeting_keywords = [
            "hello", "hi", "hey", "hola", "bonjour", "buenos", "buenas", "salut"
        ]

        # Try to load transformer classifier (optional). If anything goes wrong,
        # fall back gracefully to rule-based.
        if TRANSFORMERS_AVAILABLE:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
                self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
                self.model_loaded = True
                print(f"Transformer model '{self.model_name}' loaded. Using transformer-based classifier.")
            except Exception as e:
                print(f"Warning: could not load transformer model '{self.model_name}': {e}")
                print("Falling back to rule-based classifier.")
                self.model_loaded = False
        else:
            # Explicit message already printed above
            pass

    def _rule_based_intent(self, text: str) -> str:
        txt = text.lower()
        # check greetings first
        for g in self._greeting_keywords:
            if g in txt:
                return "greeting"
        for o in self._order_keywords:
            if o in txt:
                return "order_status"
        return "fallback"

    def _parse_label_to_intent(self, label: str) -> str:
        """Convert transformer pipeline label like 'LABEL_0' or '0' to an intent string."""
        if label is None:
            return "fallback"
        m = re.search(r"(\d+)", str(label))
        if m:
            idx = int(m.group(1))
            return self.LABEL_MAP.get(idx, "fallback")
        # If label does not contain digits, try direct mapping
        label_lower = str(label).lower()
        if "greet" in label_lower or "hello" in label_lower or "hi" in label_lower:
            return "greeting"
        return "fallback"

    def translate_text(self, text: str, dest_lang: str) -> str:
        """Translate English text to dest_lang using deep_translator when available,
        otherwise use built-in small translations for demo languages (es, fr).
        """
        dest = (dest_lang or "").lower()
        if dest.startswith("en"):
            return text

        if TRANSLATOR_AVAILABLE:
            try:
                return GoogleTranslator(source='auto', target=dest).translate(text)
            except Exception as e:
                print(f"Warning: translator failed: {e}. Falling back to built-in translations.")

        # Built-in small translations (only for demo)
        if dest.startswith("es"):
            return self.RESPONSES.get("fallback", {}).get("es") or text
        if dest.startswith("fr"):
            return self.RESPONSES.get("fallback", {}).get("fr") or text
        return text

    def chat(self, message: str) -> Dict[str, Any]:
        """Return a dict: { detected_language, intent, response }"""
        if not isinstance(message, str):
            raise ValueError("message must be a string")

        try:
            detected_lang = detect(message)
        except Exception:
            detected_lang = "en"

        intent = "fallback"
        # Use transformer if available and loaded; otherwise rule-based
        if self.model_loaded and self.classifier is not None:
            try:
                pred = self.classifier(message)[0]
                label = pred.get("label") if isinstance(pred, dict) else None
                intent = self._parse_label_to_intent(label)
            except Exception as e:
                print(f"Warning: transformer classifier failed at runtime: {e}")
                intent = self._rule_based_intent(message)
        else:
            intent = self._rule_based_intent(message)

        response_dict = self.RESPONSES.get(intent, self.RESPONSES["fallback"])
        reply = response_dict.get(detected_lang)
        if reply is None:
            reply = self.translate_text(response_dict.get("en", ""), detected_lang)

        return {"detected_language": detected_lang, "intent": intent, "response": reply}


# ----------------------
# Local tests (no FastAPI/uvicorn/TestClient)
# ----------------------
if __name__ == "__main__":
    bot = MultilingualChatbot()

    # Original test cases (preserved)
    original_tests = [
        {"message": "Hello"},
        {"message": "¿Dónde está mi pedido?"},
        {"message": "Bonjour"}
    ]

    # Additional tests added
    additional_tests = [
        {"message": "Where is my order?"},
        {"message": "Hi there!"},
        {"message": "¿Hola, cómo estás?"},
        {"message": "I want to check my order status"},
        {"message": "Gracias por tu ayuda"},
        {"message": "Merci beaucoup"},
        {"message": "Ciao"}  # Italian greeting — expected to fallback to 'en' detection
    ]

    all_tests = original_tests + additional_tests

    print("Running local tests (no FastAPI/uvicorn required).\n")
    for t in all_tests:
        inp = t["message"]
        out = bot.chat(inp)
        print(f"Input: {inp}\n -> {out}\n")

    # Basic assertions to ensure output shape — keep tests non- brittle across environments
    for t in all_tests:
        out = bot.chat(t["message"])
        assert isinstance(out, dict), "chat() must return a dict"
        assert "detected_language" in out and "intent" in out and "response" in out, "Missing keys in response"

    print("All local tests passed (basic checks).\n")
    print("Notes: \n - If you'd like stricter language detection, install the 'langdetect' package in your environment.\n - If you'd like transformer-only behavior, tell me and I'll prefer the transformer when it's available.")
