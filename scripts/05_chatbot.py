import sys
import os
import json
import pickle
import argparse
import numpy as np
from scipy.stats import entropy as scipy_entropy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "env"))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

import torch
from environment import environment
from agent import Policy_Gradient_pair_model
from llm_interface import LLMInterface


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return argparse.Namespace(**{
        k: v for k, v in cfg.items()
        if k not in ("description", "comments")
    })


def update_state_for_symptom(env, symp_idx, data_type, props, raw_val):
    """
    Injecte la valeur `raw_val` dans env.target_state pour le symptôme `symp_idx`.
    Retourne la valeur réellement injectée ou None si erreur.
    """
    try:
        mask = env.action_mask[symp_idx] == 1
        if not mask.any():
            return None  # symptôme déjà interrogé ou hors-masque

        if data_type == "B":
            v = int(raw_val) if raw_val in (1, -1, 0) else 0
            env.target_state[0, mask] = v
            return v

        elif data_type == "C":
            props_vals = [str(p["value"]).lower() for p in props[:-1]] if props else []
            num = len(props_vals)
            if raw_val == 0 or str(raw_val) == "0" or num == 0:
                env.target_state[0, mask] = 0.0
                return 0
            idx = next((i for i, v in enumerate(props_vals) if v == str(raw_val).lower()), 0)
            scaled = (idx + 1) / num
            env.target_state[0, mask] = scaled
            return raw_val

        elif data_type == "M":
            if raw_val == 0 or str(raw_val) == "0":
                env.target_state[0, mask] = 0.0
                return 0
            props_vals = [str(p["value"]).lower() for p in props[:-1]] if props else []
            idx = next((i for i, v in enumerate(props_vals) if v == str(raw_val).lower()), 0)
            mapping = env.symptom_to_obs_mapping.get(symp_idx)
            if mapping and len(mapping) > 0:
                frame_idx = mapping[0] + idx
                env.target_state[0, :] = -1
                env.target_state[0, frame_idx] = 1.0
            else:
                env.target_state[0, mask] = 0.0
            return raw_val

    except Exception as e:
        print(f"\n[ Erreur état ({symp_idx})] {e} → on passe à 0.0")
        try:
            env.target_state[0, env.action_mask[symp_idx] == 1] = 0.0
        except Exception:
            pass
        return None


def ask_initial_symptoms(llm: LLMInterface, env: environment) -> list[int]:
    """
    Demande à l'utilisateur de décrire ses symptômes initiaux en texte libre.
    Retourne la liste des indices de symptômes détectés dans le catalogue.
    """
    print("\n" + "=" * 60)
    print(" DÉCRIVEZ VOS SYMPTÔMES INITIAUX")
    print("=" * 60)
    print("Décrivez en quelques mots pourquoi vous consultez aujourd'hui.")
    print("Exemple : « J'ai de la fièvre et mal à la tête depuis hier. »")
    print("(Laissez vide pour utiliser un patient aléatoire du jeu de test)")
    print("-" * 60)

    user_input = input("Vous : ").strip()
    if not user_input:
        return []

    symp_keys = llm.extract_initial_symptoms(user_input)

    found_indices = []
    for key in symp_keys:
        idx = env.symptom_name_2_index.get(key)
        if idx is not None:
            found_indices.append(idx)

    if found_indices:
        names = [env.symptom_index_2_key[i] for i in found_indices]
        print(f"\n Symptômes reconnus : {', '.join(names)}")
        llm.add_to_history("user", user_input)
        llm.add_to_history("assistant", f"Compris. Vous présentez : {', '.join(names)}.")
    else:
        print("\n Aucun symptôme du catalogue reconnu. Démarrage avec un patient aléatoire.")

    return found_indices


def main():
    print("=" * 60)
    print(" CHATBOT MÉDICAL AARLC (Conversationnel LLM)")
    print("=" * 60)

    config_path = os.path.join(BASE_DIR, "configs", "config_maladies_fr.json")
    args = load_config(config_path)
    args.train = False

    data_dir = os.path.join(BASE_DIR, "data", "processed")
    models_dir = os.path.join(BASE_DIR, "output", "models")
    evi_meta = os.path.join(data_dir, "release_evidences.json")
    patho_meta = os.path.join(data_dir, "release_conditions.json")
    args.evi_meta_path = evi_meta
    args.patho_meta_path = patho_meta

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(models_dir, "best_model.pkl")
    threshold_path = os.path.join(models_dir, "threshold.pkl")

    if not os.path.exists(model_path):
        print(f" Modèle introuvable : {model_path}")
        sys.exit(1)

    with open(evi_meta, "r", encoding="utf-8") as f:
        evidences_data = json.load(f)

    test_path = os.path.join(data_dir, "test.csv")
    env = environment(args, test_path, train=False)

    print("\n Connexion à Ollama (phi4-mini)...")
    llm = LLMInterface(args.evi_meta_path, model_name="phi4-mini")
    print("Ollama prêt.")

    agent = Policy_Gradient_pair_model(
        state_size=env.state_size,
        disease_size=env.diag_size,
        symptom_size=env.symptom_size,
        device=device,
    )
    agent.load(model_path)

    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
    else:
        threshold = args.threshold * np.ones(env.diag_size)

    # ── Demander les symptômes initiaux à l'utilisateur ──────────────────
    initial_symptom_indices = ask_initial_symptoms(llm, env)

    # ── Chargement d'un patient du test set ──────────────────────────────
    import random
    patient_id = random.randint(0, env.sample_size - 1)
    env.reset()
    env.idx = patient_id

    s, disease, diff_indices, diff_probas, _ = env.initialize_state(1)
    true_disease_idx = disease[0]
    true_disease_name = env.pathology_index_2_key[true_disease_idx]

    # ── Si l'utilisateur a fourni des symptômes, on les injecte dans l'état
    if initial_symptom_indices:
        first_symp_idx = initial_symptom_indices[0]
        frame_idx = env.symptom_to_obs_mapping[first_symp_idx][0]
        s[0, frame_idx] = 1
        env.inquired_symptoms[0, first_symp_idx] = 1

        # On marque les autres symptômes cités dans target_state
        for si in initial_symptom_indices[1:]:
            sym_key = env.symptom_index_2_key[si]
            ev = evidences_data.get(sym_key, {})
            props = ev.get("propositions_fr", [])
            update_state_for_symptom(env, si, ev.get("data_type", "B"), props, 1)
            env.inquired_symptoms[0, si] = 1

        symp_names = ", ".join(env.symptom_index_2_key[i] for i in initial_symptom_indices)
        intro = f"Vous consultez pour : {symp_names}"
    else:
        # Fallback : utiliser le symptôme initial du patient simulé
        initial_symp_idx = np.where(env.inquired_symptoms[0] == 1)[0]
        if len(initial_symp_idx) > 0:
            symp_key = env.symptom_index_2_key[initial_symp_idx[0]]
            intro = f"Vous consultez pour : {symp_key.replace('_', ' ')}"
        else:
            intro = "Bonjour, comment puis-je vous aider ?"

    done = np.zeros(1, dtype=bool)
    right_diagnosis = np.zeros(1, dtype=bool)

    a_d_0, p_d_0 = agent.choose_diagnosis(s)
    ent_init = scipy_entropy(p_d_0, axis=1)
    ent = ent_init.copy()
    ent_init[ent_init < 1e-8] = 1.0

    print(f"\n Bonjour ! Je suis l'assistant médical virtuel.")
    print(f"   (Patient test n°{patient_id} — maladie cachée : {true_disease_name})")
    print(f"\n>>> {intro}")
    llm.add_to_history("assistant", intro)

    # ── Boucle de consultation ────────────────────────────────────────────
    for step in range(args.MAXSTEP):
        # Vérification pré-diagnostic
        a_d_, p_d_ = agent.choose_diagnosis(s)
        ent_curr = scipy_entropy(p_d_, axis=1)
        if ent_curr[0] < threshold[a_d_[0]]:
            print("\n" + "-" * 40)
            print(" J'ai suffisamment d'informations pour établir un diagnostic.")
            break

        # Choix du prochain symptôme par l'agent RL
        a_p, _, _ = agent.choose_action_s(s)
        symp_idx = a_p[0]
        symp_key = env.symptom_index_2_key[symp_idx]
        ev_data = evidences_data.get(symp_key, {})
        data_type = ev_data.get("data_type", "B")
        props = ev_data.get("propositions_fr", [
            {"label": "Oui", "value": 1},
            {"label": "Non", "value": -1},
            {"label": "Ne sait pas", "value": 0},
        ])

        print(f"\n{'~'*40}")
        print(f"[AARLC] Symptôme interrogé : \033[1m{symp_key}\033[0m  (type={data_type})")

        # NLG : génération en streaming (print interne à generate_question)
        natural_q = llm.generate_question(symp_key)

        raw_val = None
        while raw_val is None:
            user_input = input(" Vous : ").strip()
            if not user_input:
                continue

            llm.add_to_history("user", user_input)

            # NLU
            extracted = llm.extract_information(user_input, symp_key)
            intent = extracted.get("intent", "answer_question")
            symp_answers = extracted.get("symptoms", [])

            # ── DEBUG : afficher ce que le LLM a extrait ──
            print(f"  [DEBUG NLU] intent={intent!r}  symptoms={symp_answers}")

            if intent == "ask_clarification":
                llm.generate_clarification(symp_key, user_input)
                continue

            current_ans = next(
                (s for s in symp_answers if s.get("key") == symp_key), None
            )

            if current_ans is not None:
                val = current_ans.get("value", 0)
                if isinstance(val, str) and val.lstrip("-").isdigit():
                    val = int(val)
                raw_val = val
                print(f"  [DEBUG] Valeur retenue pour « {symp_key} » → {raw_val}")
            elif symp_answers:
                # D'autres symptômes mentionnés → on les note en passant
                for extra in symp_answers:
                    ek = extra.get("key")
                    ev = extra.get("value", 0)
                    ei = env.symptom_name_2_index.get(ek)
                    if ei is not None:
                        eev = evidences_data.get(ek, {})
                        update_state_for_symptom(env, ei, eev.get("data_type", "B"),
                                                  eev.get("propositions_fr", []), ev)
                        env.inquired_symptoms[0, ei] = 1
                print(f"  [DEBUG] Symptômes annexes notés : {[a['key'] for a in symp_answers]}")
                print(f" Agent : Noté. Mais pourriez-vous aussi me dire : {natural_q}")
            else:
                print(f" Agent : Je n'ai pas compris. Pouvez-vous répondre à : {natural_q}")

        # Mise à jour de l'état pour le symptôme principal
        injected = update_state_for_symptom(env, symp_idx, data_type, props, raw_val)
        print(f"  [DEBUG] Injection dans env.target_state[{symp_idx}] = {injected}")

        # Avancer l'environnement
        s, _, done_batch, right_diagnosis, diag, ent, a_d_ = env.step(
            s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent
        )
        done = done_batch

        if done[0]:
            print("\n" + "-" * 40)
            print(" Diagnostic établi avec confiance suffisante.")
            break

    # ── Diagnostic final ──────────────────────────────────────────────────
    a_d_final, p_d_final = agent.choose_diagnosis(s)
    pred_idx = a_d_final[0]
    pred_name = env.pathology_index_2_key[pred_idx]
    confidence = p_d_final[0, pred_idx] * 100

    print("\n" + "=" * 60)
    print(" BILAN DE LA CONSULTATION")
    print("=" * 60)
    print(f"Diagnostic prédit : {pred_name} (Confiance : {confidence:.1f}%)")
    print(f"Vraie pathologie  : {true_disease_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
