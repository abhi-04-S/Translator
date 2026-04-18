from fastapi import FastAPI
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

# 🚨 Emergency mapping
EMERGENCY_MAP = {
    "my life is in danger": "DANGER LIFE MY",
    "i am in danger": "DANGER I",
    "help me": "HELP ME",
    "save me": "SAVE ME",
    "there is fire": "FIRE",
    "call police": "POLICE CALL",
}

# 🧠 Time words
TIME_WORDS = ["today", "tomorrow", "yesterday", "now"]

# ❓ Question words
QUESTION_WORDS = ["what", "where", "why", "when", "how", "who"]


@app.post("/process")
def process(data: dict):
    text = data.get("text", "").lower().strip()

    # 🚨 1. Emergency shortcut
    if text in EMERGENCY_MAP:
        return build_response(text, EMERGENCY_MAP[text])

    doc = nlp(text)

    subject = ""
    verb = ""
    obj = ""
    extra = ""
    negation = False
    time_context = ""
    is_question = False

    # 🧠 2. NLP Parsing
    for token in doc:

        # 👤 Subject
        if "subj" in token.dep_:
            subject = normalize_pronoun(token.text)

        # ⚡ Verb
        elif token.dep_ == "ROOT":
            verb = token.lemma_.upper()

        # 🎯 Object (accurate)
        elif "obj" in token.dep_ or token.dep_ == "pobj":
            obj = token.text.upper()

        # 🔥 Fallback object detection
        if (
            not obj
            and token.pos_ == "NOUN"
            and token.text not in TIME_WORDS
        ):
            obj = token.text.upper()

        # ❌ Negation
        if token.dep_ == "neg":
            negation = True

        # ⏰ Time
        if token.text in TIME_WORDS:
            time_context = token.text.upper()

        # ⚠️ Extra conditions
        if token.text in ["danger", "fire", "help", "accident"]:
            extra = token.text.upper()

        # ❓ Question
        if token.text in QUESTION_WORDS:
            is_question = True

    # ❓ Question by punctuation
    if "?" in text:
        is_question = True

    # 🔄 3. Build ISL structure
    isl_parts = []

    # Priority: Time → Condition → Object → Subject → Verb
    if time_context:
        isl_parts.append(time_context)

    if extra:
        isl_parts.append(extra)

    if obj:
        isl_parts.append(obj)

    if subject:
        isl_parts.append(subject)

    if verb:
        isl_parts.append(verb)

    if negation:
        isl_parts.append("NOT")

    if is_question:
        isl_parts.append("QUESTION")

    # 🛟 4. Fallback
    if not isl_parts:
        isl_parts = [token.text.upper() for token in doc if token.is_alpha]

    # 🔥 Remove duplicates while preserving order
    isl_parts = list(dict.fromkeys(isl_parts))

    isl = " ".join(isl_parts)

    return build_response(text, isl)


# 🧩 Normalize pronouns
def normalize_pronoun(word):
    mapping = {
        "i": "I",
        "me": "ME",
        "my": "MY",
        "you": "YOU",
        "he": "HE",
        "she": "SHE",
        "we": "WE",
        "they": "THEY"
    }
    return mapping.get(word, word.upper())


# 📦 Response format
def build_response(text, isl):
    return {
        "input": text,
        "isl": isl,
        "tokens": isl.split(" ")
    }