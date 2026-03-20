"""
scripts/llm_interface.py
Interface Ollama (phi4-mini) optimisée pour la vitesse.

Stratégie :
- NLU  : streaming + prompt ultra-court → réponse sur 1 ligne (OUI/NON/INCONNU/VALEUR)
         interprétée côté Python par regex → évite le format=json lent (82s→<15s)
- NLG  : streaming → tokens affichés au fur et à mesure
- Mémoire : 5 derniers tours (user/assistant) injectés dans NLG uniquement
"""
import json
import re
import sys
import urllib.request


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de décodage des réponses texte NLU
# ──────────────────────────────────────────────────────────────────────────────
_YES_RE = re.compile(
    r"\b(oui|yes|présent|positif|j'en souffre|en ai|j'ai|affirmatif|correct|c'est ça)\b",
    re.I,
)
_NO_RE = re.compile(
    r"\b(non|no|absent|négatif|pas|aucun|jamais|absolument pas)\b",
    re.I,
)
_UNSURE_RE = re.compile(
    r"\b(sait pas|inconnu|peut-être|je (ne )?sais pas|incertain|pas sûr|incertitude)\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\b(pourquoi|c'est quoi|qu'est-ce|comment|expliqu|clarifie|je ne comprends|"
    r"signifie|c'est quoi|pas compris|pas clair)\b",
    re.I,
)


def _decode_boolean_answer(text: str) -> int | None:
    """Retourne 1, -1 ou 0 depuis une réponse texte libre. None si ambigu."""
    t = text.strip().lower()
    if _QUESTION_RE.search(t):
        return None  # → ask_clarification
    if _YES_RE.search(t):
        return 1
    if _NO_RE.search(t):
        return -1
    if _UNSURE_RE.search(t):
        return 0
    return None  # ambigu


# ──────────────────────────────────────────────────────────────────────────────
# Classe principale
# ──────────────────────────────────────────────────────────────────────────────
class LLMInterface:
    MEMORY_WINDOW = 5  # paires (question/réponse) conservées

    def __init__(self, evidences_path: str, model_name: str = "phi4-mini"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/chat"
        self.history: list = []

        with open(evidences_path, "r", encoding="utf-8") as f:
            self.raw_evidences = json.load(f)

        # Index rapide clé → métadonnées
        self.symptom_keys = list(self.raw_evidences.keys())

    # ── Mémoire ──────────────────────────────────────────────────────────── #
    def add_to_history(self, role: str, text: str):
        text = text.strip()
        if not text:
            return
        self.history.append({"role": role, "content": text})
        max_msgs = self.MEMORY_WINDOW * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def _safe_history(self) -> list:
        valid = {"user", "assistant"}
        return [m for m in self.history
                if m.get("role") in valid and m.get("content", "").strip()]

    # ── Construction messages Ollama ──────────────────────────────────────── #
    def _build_messages(self, system: str, user: str,
                        inject_history: bool = False) -> list:
        msgs = [{"role": "system", "content": system}]
        if inject_history:
            msgs.extend(self._safe_history())
        msgs.append({"role": "user", "content": user})
        return msgs

    # ── Appel streaming générique ────────────────────────────────────────── #
    def _stream(self, system: str, user: str,
                max_tokens: int = 80,
                inject_history: bool = False,
                print_output: bool = True) -> str:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(system, user, inject_history),
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        full = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        if print_output:
                            print(token, end="", flush=True)
                        full += token
                    if chunk.get("done", False):
                        break
        except Exception as exc:
            print(f"\n[LLM Stream Error] {exc}", file=sys.stderr)

        if print_output:
            print()  # saut de ligne
        return full.strip()

    # ── NLU – Détection intention ──────────────────────────────────────────── #
    def detect_intent(self, user_text: str) -> str:
        """
        Détecte rapidement si le patient : répond, pose une question, ou décrit des symptômes.
        Retourne 'answer_question' | 'ask_clarification' | 'provide_symptom'
        """
        # Heuristique rapide côté Python SANS appel LLM (pour la vitesse)
        t = user_text.strip()
        if _QUESTION_RE.search(t):
            return "ask_clarification"
        if _YES_RE.search(t) or _NO_RE.search(t) or _UNSURE_RE.search(t):
            return "answer_question"
        # Cas ambigu : on considère que c'est une réponse à la question
        return "answer_question"

    # ── NLU – Extraction valeur booléenne (rapide via streaming) ──────────── #
    def extract_boolean(self, user_text: str, symptom_key: str) -> int | None:
        """
        Extrait la valeur Booléen (1/-1/0) de la réponse du patient pour le symptôme_key.
        Utilise d'abord l'heuristique regex (rapide).
        Si ambigu, appelle phi4-mini avec un prompt ultra-court en streaming.
        Retourne None si intention = clarification.
        """
        # 1) Essai rapide regex
        if _QUESTION_RE.search(user_text):
            return None  # → ask_clarification

        val = _decode_boolean_answer(user_text)
        if val is not None:
            return val

        # 2) Fallback LLM uniquement si ambigu (prompt minimaliste)
        system = (
            "Tu es un extracteur médical. "
            "Réponds avec UN SEUL mot : OUI, NON ou INCONNU."
        )
        user = f"Le patient dit: \"{user_text}\". A-t-il le symptôme « {symptom_key} » ?"

        response = self._stream(system, user, max_tokens=5,
                                print_output=False)
        r = response.upper()
        if "OUI" in r:
            return 1
        if "NON" in r:
            return -1
        return 0

    # ── NLU – Extraction valeur catégorielle ─────────────────────────────── #
    def extract_categorical(self, user_text: str, symptom_key: str,
                            options: list) -> str | int | None:
        """
        Extrait la valeur catégorielle parmi `options` pour le symptôme_key.
        Retourne None si clarification.
        """
        if _QUESTION_RE.search(user_text):
            return None

        # Vérifie d'abord si une option est mentionnée textuellement dans la réponse
        t = user_text.lower()
        for opt in options:
            if str(opt).lower() in t:
                return opt

        # Fallback LLM ultra-court
        opts_str = " | ".join(str(o) for o in options)
        system = f"Réponds avec UN SEUL mot parmi : {opts_str} | INCONNU"
        user = f"Le patient dit: \"{user_text}\". Quelle est la valeur pour « {symptom_key } » ?"

        response = self._stream(system, user, max_tokens=8, print_output=False)
        r = response.strip().lower()
        for opt in options:
            if str(opt).lower() in r:
                return opt
        return 0  # INCONNU

    # ── NLU – Point d'entrée unifié ───────────────────────────────────────── #
    def extract_information(self, user_text: str,
                            current_symptom: str | None = None) -> dict:
        """
        Analyse le texte patient.
        Retourne {"intent": str, "symptoms": [{"key": str, "value": ...}]}
        """
        intent = self.detect_intent(user_text)

        if intent == "ask_clarification" or current_symptom is None:
            return {"intent": intent, "symptoms": []}

        # On détermine le type du symptôme actuel
        ev = self.raw_evidences.get(current_symptom, {})
        data_type = ev.get("data_type", "B")

        if data_type == "B":
            val = self.extract_boolean(user_text, current_symptom)
        else:
            options = [p["value"] for p in ev.get("propositions_fr", [])
                       if p.get("value") not in (0, None)]
            val = self.extract_categorical(user_text, current_symptom, options)

        if val is None:
            # extract_boolean retourne None = ask_clarification
            return {"intent": "ask_clarification", "symptoms": []}

        return {
            "intent": "answer_question",
            "symptoms": [{"key": current_symptom, "value": val}]
        }

    # ── NLG – Question empathique (streaming) ─────────────────────────────── #
    def generate_question(self, symptom_key: str) -> str:
        ev = self.raw_evidences.get(symptom_key, {})
        question_fr = ev.get("question_fr", f"Avez-vous : {symptom_key} ?")
        ctx = "antécédent" if ev.get("is_antecedent") else "symptôme"

        system = (
            f"Tu es un médecin empathique. Reformule cette question sur un {ctx} "
            "de façon naturelle et chaleureuse en UNE phrase courte. "
            "Réponds seulement par la question, sans guillemets ni introduction."
        )
        user = f"Question: {question_fr}"

        print("👨‍⚕️ Agent : ", end="", flush=True)
        result = self._stream(system, user, max_tokens=50,
                              inject_history=False)
        self.add_to_history("assistant", result)
        return result

    # ── NLG – Clarification ───────────────────────────────────────────────── #
    def generate_clarification(self, symptom_key: str,
                               user_question: str) -> str:
        system = (
            "Tu es un médecin pédagogue. "
            "Explique le terme en UNE PHRASE simple (max 15 mots). "
            "Pas de jargon médical."
        )
        user = f"Terme: « {symptom_key} ». Question: « {user_question} »"

        print("👨‍⚕️ Agent : ", end="", flush=True)
        result = self._stream(system, user, max_tokens=40,
                              inject_history=False)
        return result
