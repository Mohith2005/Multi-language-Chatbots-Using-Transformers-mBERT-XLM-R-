# Multi-language-Chatbots-Using-Transformers-mBERT-XLM-R-
This project demonstrates a **Multilingual Chatbot** capable of understanding user input in multiple languages (English, Spanish, French) and responding with context-aware answers. The project addresses the challenge of global customer engagement by using transformer models (mBERT/XLM-R) when available, or rule-based intent detection as a fallback.

**Objective:**
- Build a chatbot that can detect language, classify user intent (greeting, order status, fallback), and generate responses in the same language.
- Showcase how multilingual conversational AI can scale global support while maintaining natural communication.

**Tech Stack:**
- **Python** for core logic.
- **Transformers** library (Hugging Face) for multilingual intent classification.
- **Deep Translator** for automatic language translation fallback.
- Heuristic language detection fallback (when `langdetect` is unavailable).

**Expected Outcome:**
- Smoothly detect language and intent from user messages.
- Respond in matching language with predefined templates or translations.
- Demonstrate extensibility for enterprise-grade multilingual AI solutions.


Multilingual Chatbot (FastAPI-free) — langdetect fallback

This file is a rewrite that avoids requiring external runtime dependencies
that may be unavailable in sandboxed environments (e.g. `ssl` or missing
`langdetect`). It provides:
 - A `MultilingualChatbot` class with a `chat(message: str) -> dict` method.
 - An attempt to use `transformers` for intent classification if available.
 - A robust fallback **language detector** implemented with heuristics when
   the `langdetect` package is not installed.
 - Optional use of `deep_translator.GoogleTranslator` when available; otherwise
   small built-in translations are used for demo purposes.
 - Self-contained local tests (no FastAPI/uvicorn/TestClient) that run when
   executing this file. Original test cases are preserved and additional
   test cases are included as requested.

If you want different behavior (for example: "always use transformer when
available" vs "always prefer rule-based"), tell me which behavior you want
and I'll update the file accordingly.
**
