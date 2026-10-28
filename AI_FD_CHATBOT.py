"""
BA3.2 AI Solution
Chatbot (softbot) Component

File:
    AI_FD_CHATBOT.py

Purpose:
    A text assistant for fraud analysts. Instead of calling Python functions,
    an analyst types plain requests and the bot scores the transaction with
    the trained model from AI_FD.py, explains the result, and answers basic
    questions about how the system works.

Who uses it:
    Fraud analysts / call-centre staff who must decide whether a flagged
    transaction is allowed, reviewed or blocked.

What it can do (intents):
    - score a transaction, either from one line of "field=value" pairs or
      through a guided question-and-answer
    - explain WHY a transaction got its score
    - explain what ALLOW / REVIEW / BLOCK mean and where the cut-offs are
    - say honestly how accurate the model is
    - list the fields it needs, and the usual fraud rate

Design decisions:
    - The score ALWAYS comes from the machine-learning model. The bot only
      does the talking, so its answers can be traced and checked.
    - Intents are recognised with simple keyword rules. They are predictable
      and need no extra libraries or internet connection.
    - Optional language layer: if a local Ollama server running Phi-3 is
      available and FD_USE_OLLAMA=1 is set, the bot's answers are passed
      through it to be reworded in friendlier language. It is told to keep
      every number and the decision unchanged. If Ollama is not running, the
      bot silently uses its own text, so nothing breaks.

How to run:
    python AI_FD_CHATBOT.py            interactive chat
    python AI_FD_CHATBOT.py --demo     scripted example conversation
"""

import contextlib
import io
import json
import os
import re
import sys
import urllib.request

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from AI_FD import (
    MODEL_FILE,
    create_directories,
    load_dataset,
    load_model,
    make_ai_decision,
    make_prediction,
    prepare_ml_data,
    preprocess_data,
    save_model,
    train_classifier
)


# CONFIGURATION

# What the analyst may type  ->  the column it means
FIELD_ALIASES = {
    "amount": "Transaction_Amount",
    "transaction_amount": "Transaction_Amount",
    "type": "Transaction_Type",
    "transaction_type": "Transaction_Type",
    "hour": "Time_of_Transaction",
    "time": "Time_of_Transaction",
    "time_of_transaction": "Time_of_Transaction",
    "device": "Device_Used",
    "device_used": "Device_Used",
    "location": "Location",
    "city": "Location",
    "prev": "Previous_Fraudulent_Transactions",
    "previous": "Previous_Fraudulent_Transactions",
    "previous_fraud": "Previous_Fraudulent_Transactions",
    "previous_fraudulent_transactions": "Previous_Fraudulent_Transactions",
    "age": "Account_Age",
    "account_age": "Account_Age",
    "txn24h": "Number_of_Transactions_Last_24H",
    "last24h": "Number_of_Transactions_Last_24H",
    "count24h": "Number_of_Transactions_Last_24H",
    "number_of_transactions_last_24h": "Number_of_Transactions_Last_24H",
    "payment": "Payment_Method",
    "method": "Payment_Method",
    "payment_method": "Payment_Method"
}

# Friendly names, also the order of the guided questions
FIELD_LABELS = {
    "Transaction_Amount": "transaction amount",
    "Transaction_Type": "transaction type",
    "Time_of_Transaction": "hour of the day",
    "Device_Used": "device used",
    "Location": "location",
    "Previous_Fraudulent_Transactions": "previous fraudulent transactions",
    "Account_Age": "account age",
    "Number_of_Transactions_Last_24H": "transactions in the last 24 hours",
    "Payment_Method": "payment method"
}

# Wording of the guided questions
QUESTIONS = {
    "Transaction_Amount": "What is the transaction amount?",
    "Transaction_Type": "What type of transaction is it?",
    "Time_of_Transaction": "At what hour of the day (0-23) did it happen?",
    "Device_Used": "Which device was used?",
    "Location": "Where did it happen?",
    "Previous_Fraudulent_Transactions": "How many previous fraudulent transactions does this customer have?",
    "Account_Age": "How old is the account?",
    "Number_of_Transactions_Last_24H": "How many transactions has the account made in the last 24 hours?",
    "Payment_Method": "Which payment method was used?"
}

TIME_FIELD = "Time_of_Transaction"

# Pairs like  amount=950  or  type=online purchase  (the value may have spaces)
PAIR_PATTERN = re.compile(r"([A-Za-z_0-9]+)\s*=\s*(.+?)(?=\s+[A-Za-z_0-9]+\s*=|$)")

BLANK_WORDS = ("", "skip", "unknown", "none", "n/a", "blank", "missing")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"


# OPTIONAL LANGUAGE LAYER (Ollama / Phi-3)

def polish_with_llm(text, model_name=OLLAMA_MODEL, url=OLLAMA_URL, timeout=20):
    """
    Ask a local Phi-3 model (through Ollama) to reword the bot's answer in
    friendlier plain English. The model is told not to change any number or
    the decision, and it never produces the fraud score itself.

    If Ollama is not running, or anything goes wrong, the original text is
    returned unchanged.
    """
    prompt = (
        "Rewrite the following message from a fraud-detection assistant in "
        "clear, friendly plain English for a bank analyst. Keep every number "
        "and the decision (ALLOW, REVIEW or BLOCK) exactly as written. Do not "
        "add any new facts.\n\n" + text
    )
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read().decode("utf-8")).get("response", "").strip()
        return answer or text
    except Exception:
        return text


# THE CHATBOT

class FraudChatbot:
    """
    Keyword-driven assistant around a trained fraud model.

    model:             fitted pipeline from AI_FD.py (preprocessing + model)
    reference:         DataFrame of normal feature values (training data),
                       used for allowed categories and "typical" values
    review_threshold:  score at or above which a transaction is REVIEWed
    block_threshold:   score at or above which a transaction is BLOCKed
    metrics:           dict with pr_auc and fraud_rate on held-out data
    use_llm:           reword answers through Ollama/Phi-3 when available
    """

    def __init__(self, model, reference, review_threshold, block_threshold,
                 metrics=None, use_llm=False):
        self.model = model
        self.review_threshold = review_threshold
        self.block_threshold = block_threshold
        self.metrics = metrics or {}
        self.use_llm = use_llm

        self.numeric_fields = set(reference.select_dtypes(include=np.number).columns)
        self.categories = {
            c: sorted(reference[c].dropna().unique())
            for c in reference.columns if c not in self.numeric_fields
        }
        self.typical = {
            c: (reference[c].median() if c in self.numeric_fields else reference[c].mode()[0])
            for c in reference.columns
        }

        self.last = None      # the most recent scored transaction
        self.pending = None   # fields still to ask about in guided mode
        self.collected = {}

    # ---- public entry point -------------------------------------------

    def greeting(self):
        return ("Hello! I'm the fraud assistant. I can score a transaction, explain "
                "why it got its score, and explain what ALLOW, REVIEW and BLOCK mean. "
                "Type 'help' to see everything I can do.")

    def reply(self, message):
        """Return the bot's answer to one message from the analyst."""
        answer = self._route(message.strip())

        # Rewording is only applied to answers, not to the guided questions
        # (pending is still set while the bot is waiting for more answers).
        if self.use_llm and self.pending is None:
            answer = polish_with_llm(answer)

        return answer

    # ---- routing -------------------------------------------------------

    def _route(self, text):
        lowered = text.lower()

        if self.pending is not None:
            return self._guided_step(text)

        if PAIR_PATTERN.search(text):
            return self._score_from_pairs(text)

        first_word = lowered.split()[0] if lowered.split() else ""
        if first_word in ("hi", "hello", "hey") or lowered.startswith("good morning"):
            return self.greeting()
        if "help" in lowered or "what can you do" in lowered:
            return self._help()
        if any(w in lowered for w in ("why", "explain", "reason", "how did you")):
            return self._explain()
        if any(w in lowered for w in ("threshold", "review", "block", "allow", "decision", "mean")):
            return self._tiers()
        if any(w in lowered for w in ("accura", "reliable", "trust", "performance", "how well", "how good")):
            return self._reliability()
        if any(w in lowered for w in ("how common", "fraud rate", "how often", "base rate")):
            return self._base_rate()
        if any(w in lowered for w in ("field", "column", "what do you need", "input")):
            return self._fields_help()
        if any(w in lowered for w in ("score", "check", "predict", "assess", "transaction")):
            return self._start_guided()

        return ("I'm not sure what you mean. I can score a transaction, explain a score, "
                "explain ALLOW / REVIEW / BLOCK, or tell you how accurate I am. Type 'help' for examples.")

    # ---- scoring -------------------------------------------------------

    def _blank_fields(self):
        return {name: None for name in FIELD_LABELS}

    def _score_from_pairs(self, text):
        fields = self._blank_fields()
        notes, ignored = [], []

        for key, value in PAIR_PATTERN.findall(text):
            field = FIELD_ALIASES.get(key.lower())
            if field is None:
                ignored.append(key)
                continue
            parsed, problem, note = self._parse_value(field, value)
            if problem:
                return f"I couldn't use {FIELD_LABELS[field]}: {problem} Please try again."
            fields[field] = parsed
            if note:
                notes.append(note)

        answer = self._score(fields)
        if ignored:
            answer += f"\n(I ignored unknown fields: {', '.join(ignored)}.)"
        if notes:
            answer += "\n" + "\n".join(notes)
        return answer

    def _score(self, fields):
        probability = make_prediction(self.model, fields)
        decision = make_ai_decision(
            probability, self.review_threshold, self.block_threshold
        )["decision"]
        self.last = {"fields": fields, "probability": probability, "decision": decision}

        rate = self.metrics.get("fraud_rate")
        lines = [f"Fraud score: {probability:.3f} ({probability * 100:.1f}% estimated chance of fraud)."]
        if rate:
            lines[0] = lines[0][:-1] + f", {probability / rate:.1f}x the usual fraud rate of {rate * 100:.1f}%."
        lines.append(f"Decision: {decision}")

        used = [f"{FIELD_LABELS[k]} {v:g}" if isinstance(v, float) else f"{FIELD_LABELS[k]} {v}"
                for k, v in fields.items() if v is not None]
        missing = [FIELD_LABELS[k] for k, v in fields.items() if v is None]
        lines.append("Used: " + ("; ".join(used) if used else "nothing"))
        if missing:
            lines.append("Not provided (treated as missing): " + ", ".join(missing))
        lines.append("Ask me 'why' for an explanation.")
        return "\n".join(lines)

    # ---- parsing -------------------------------------------------------

    def _parse_value(self, field, text):
        """Return (value, problem, note). problem is set when the value is unusable."""
        text = text.strip()
        if text.lower() in BLANK_WORDS:
            return None, None, None

        if field in self.numeric_fields:
            if field == TIME_FIELD and ":" in text:
                text = text.split(":")[0]  # "14:30" -> hour 14
            try:
                number = float(text.replace(",", ""))
            except ValueError:
                return None, f"'{text}' is not a number.", None
            if number < 0:
                return None, "it cannot be negative.", None
            if field == TIME_FIELD and number > 23:
                return None, "the hour must be between 0 and 23.", None
            return number, None, None

        known = {c.lower(): c for c in self.categories[field]}
        if text.lower() in known:
            return known[text.lower()], None, None
        return text, None, (f"Note: '{text}' was not seen in the training data for "
                            f"{FIELD_LABELS[field]}, so it is treated as an unknown value.")

    # ---- guided mode ---------------------------------------------------

    def _start_guided(self):
        self.pending = list(FIELD_LABELS)
        self.collected = {}
        return ("Let's check a transaction. Answer each question, or type 'skip' if you "
                "don't know. Type 'cancel' to stop.\n" + self._question(self.pending[0]))

    def _question(self, field):
        question = QUESTIONS[field]
        if field in self.categories:
            question += " Options: " + ", ".join(self.categories[field]) + "."
        return question

    def _guided_step(self, text):
        if text.lower() in ("cancel", "stop", "quit"):
            self.pending, self.collected = None, {}
            return "Cancelled. Ask me anything else, or type 'score' to start again."

        field = self.pending[0]
        value, problem, note = self._parse_value(field, text)
        if problem:
            return f"Sorry, {problem} " + self._question(field)

        self.collected[field] = value
        self.pending.pop(0)

        if self.pending:
            return (note + "\n" if note else "") + self._question(self.pending[0])

        fields, self.collected, self.pending = self.collected, {}, None
        return self._score({**self._blank_fields(), **fields})

    # ---- explanations --------------------------------------------------

    def _explain(self):
        if self.last is None:
            return "There is nothing to explain yet. Give me a transaction first, for example: score amount=950 hour=2"

        fields, probability = self.last["fields"], self.last["probability"]

        # How much does each given value move the score compared with a typical value?
        shifts = []
        for name, value in fields.items():
            if value is None:
                continue
            altered = dict(fields)
            altered[name] = self.typical[name]
            shifts.append((name, probability - make_prediction(self.model, altered)))
        shifts.sort(key=lambda s: abs(s[1]), reverse=True)

        lines = [
            f"This transaction scored {probability:.3f} and was marked {self.last['decision']} "
            f"(REVIEW from {self.review_threshold:.3f}, BLOCK from {self.block_threshold:.3f}).",
        ]
        if shifts:
            lines.append("Values that moved the score most, compared with a typical value:")
            for name, shift in shifts[:3]:
                direction = "raised" if shift > 0 else "lowered"
                lines.append(f"  - {FIELD_LABELS[name]} {direction} it by {abs(shift):.3f}")
        missing = [FIELD_LABELS[k] for k, v in fields.items() if v is None]
        if missing:
            lines.append("Missing fields (" + ", ".join(missing) + ") are treated as blanks, "
                         "which the model can read as a mild warning sign.")
        lines.append(self._reliability_note())
        return "\n".join(l for l in lines if l)

    def _tiers(self):
        return (
            "Every transaction gets a fraud score between 0 and 1, and the score decides the action:\n"
            f"  - ALLOW: below {self.review_threshold:.3f}. Let the transaction through.\n"
            f"  - REVIEW: {self.review_threshold:.3f} to {self.block_threshold:.3f}. Send to an analyst.\n"
            f"  - BLOCK: {self.block_threshold:.3f} and above. Stop it and contact the customer.\n"
            "The cut-offs are set by analyst capacity: REVIEW covers the riskiest 5% of "
            "transactions and BLOCK the riskiest 0.5%. They can be changed to suit the "
            "business's tolerance for false alarms versus missed fraud."
        )

    def _reliability(self):
        return self._reliability_note() or "I don't have accuracy figures loaded."

    def _reliability_note(self):
        pr_auc, rate = self.metrics.get("pr_auc"), self.metrics.get("fraud_rate")
        if pr_auc is None or not rate:
            return ""
        if pr_auc < rate * 1.25:
            return (f"Please note: on held-out data this model's PR-AUC is {pr_auc:.3f}, about the "
                    f"same as random guessing ({rate:.3f}). The fields in this dataset do not "
                    "separate fraud from legitimate transactions, so treat every score as "
                    "indicative only, not as proof.")
        return (f"On held-out data this model's PR-AUC is {pr_auc:.3f} against {rate:.3f} for "
                f"random guessing, which is {pr_auc / rate:.1f}x better.")

    def _base_rate(self):
        rate = self.metrics.get("fraud_rate")
        if not rate:
            return "I don't have the fraud rate loaded."
        return (f"About {rate * 100:.1f}% of transactions in the data are fraudulent, "
                f"roughly 1 in {round(1 / rate)}.")

    def _fields_help(self):
        lines = ["I use these fields (any can be left out):"]
        for name, label in FIELD_LABELS.items():
            line = f"  - {label}"
            if name in self.categories:
                line += ": " + ", ".join(self.categories[name])
            lines.append(line)
        return "\n".join(lines)

    def _help(self):
        return (
            "Here is what I can do:\n"
            "  - Score a transaction in one line, e.g.\n"
            "      score amount=950 type=online purchase hour=2 device=mobile payment=UPI\n"
            "    (other fields: location, prev, age, txn24h)\n"
            "  - Type 'score' and I'll ask you the questions one by one\n"
            "  - Type 'why' to get an explanation of the last score\n"
            "  - Ask what ALLOW, REVIEW and BLOCK mean\n"
            "  - Ask how accurate I am, or how common fraud is\n"
            "  - Type 'fields' to see what I need, or 'quit' to leave"
        )


# SETUP

def build_default_chatbot(use_llm=None):
    """
    Build the chatbot from the project's data and model. Uses the saved
    model if there is one, otherwise trains it first. The REVIEW / BLOCK
    cut-offs are set from the score distribution on held-out data.
    """
    create_directories()
    if use_llm is None:
        use_llm = os.environ.get("FD_USE_OLLAMA") == "1"

    with contextlib.redirect_stdout(io.StringIO()):  # keep the chat clean
        data = preprocess_data(load_dataset())
        X_train, X_test, y_train, y_test = prepare_ml_data(data)

        if os.path.exists(MODEL_FILE):
            model = load_model()
        else:
            model = train_classifier(X_train, y_train)
            save_model(model)

    scores = model.predict_proba(X_test)[:, 1]
    review = float(np.quantile(scores, 0.95))
    block = max(float(np.quantile(scores, 0.995)), review)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "fraud_rate": float(np.mean(y_test))
    }

    return FraudChatbot(model, X_train, review, block, metrics, use_llm)


# RUNNING

DEMO_MESSAGES = [
    "hello",
    "what can you do?",
    "score amount=950 type=online purchase hour=2 device=mobile location=Boston prev=1 age=40 txn24h=9 payment=UPI",
    "why?",
    "what do review and block mean?",
    "how accurate are you?",
    "how common is fraud?"
]


def run_demo():
    bot = build_default_chatbot(use_llm=False)
    print("Bot:", bot.greeting())
    for message in DEMO_MESSAGES:
        print(f"\nYou: {message}")
        print("\nBot:", bot.reply(message))


def run_chat():
    bot = build_default_chatbot()
    print("Bot:", bot.greeting())

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if message.lower() in ("quit", "exit", "bye"):
            print("\nBot: Goodbye.")
            break
        if not message:
            continue

        print("\nBot:", bot.reply(message))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_chat()
