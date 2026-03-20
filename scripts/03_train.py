"""
Script 03 — Entraînement AARLC
================================
Entraîne l'algorithme AARLC (REINFORCE + Classifier dual) sur le dataset maladies_fr procesé.

Architecture fidèle à AARLC (Yuan & Yu, 2021) :
- Policy Network (sym_acquire_func) : sélection de symptômes via REINFORCE
- Classifier Network (diagnosis_func) : diagnostic via soft cross-entropy
- Mécanisme adaptatif : seuil de confiance (Adaptive Alignment) via Polyak averaging

Usage:
    cd projet_maladies_fr/
    python3 scripts/03_train.py [--config configs/config_maladies_fr.json]
    python3 scripts/03_train.py --no_differential  # sans diagnostic différentiel

# ADAPTATION: Chargement depuis data/processed/ au lieu de fichiers zip DDXPlus
# ADAPTATION: Hyperparamètres adaptés à 15 conditions, 59 évidences, 1000 patients
"""

import sys
import os
import json
import argparse
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy

# Ajouter les chemins du projet au PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "env"))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

import torch
from environment import environment
from agent import Policy_Gradient_pair_model


# ─── Chargement de la config ──────────────────────────────────────────────────

def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**{
        k: v for k, v in cfg.items()
        if k not in ("description", "comments")
    })
    return args


# ─── Boucle d'évaluation ─────────────────────────────────────────────────────

def evaluate(eval_env, agent, batch_size, threshold, max_step, device):
    """Évalue l'agent sur l'intégralité du dataset de validation."""
    agent.eval()
    eval_env.reset()

    total_patients = eval_env.sample_size
    num_batches = total_patients // batch_size
    if num_batches == 0:
        num_batches = 1
        batch_size = total_patients

    total_correct = 0
    total_steps = 0
    total_done = 0

    with torch.no_grad():
        for _ in range(num_batches):
            s, disease, diff_indices, diff_probas, disease_severity = eval_env.initialize_state(batch_size)
            done = np.zeros(batch_size, dtype=bool)
            right_diagnosis = np.zeros(batch_size, dtype=bool)

            a_d_, p_d_ = agent.choose_diagnosis(s)
            ent_init = scipy_entropy(p_d_, axis=1)
            ent = ent_init.copy()
            ent_init[ent_init < 1e-8] = 1.0

            for step in range(max_step):
                a_p, _, _ = agent.choose_action_s(s)
                s, _, done, right_diagnosis, diag, ent, a_d_ = eval_env.step(
                    s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent
                )
                if np.all(done):
                    break

            # Patients non encore diagnostiqués → forcer le diagnostic
            if not np.all(done):
                a_d_final, _ = agent.choose_diagnosis(s)
                final_correct = (a_d_final == np.array(eval_env.disease)) & (~done)
                right_diagnosis |= final_correct

            total_correct += right_diagnosis.sum()
            total_steps += sum(
                step + 1 for _ in range(batch_size)  # approximation
            )
            total_done += batch_size

    accuracy = total_correct / total_done if total_done > 0 else 0.0
    avg_steps = total_steps / total_done if total_done > 0 else 0.0
    return accuracy, avg_steps


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement AARLC — Maladies FR")
    parser.add_argument("--config", type=str,
                        default=os.path.join(BASE_DIR, "configs", "config_maladies_fr.json"),
                        help="Chemin vers le fichier de configuration JSON")
    parser.add_argument("--no_differential", action="store_true",
                        help="Désactive le diagnostic différentiel (mode hard labels)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device PyTorch : 'cuda' ou 'cpu' (auto par défaut)")
    cli_args = parser.parse_args()

    # Charger la config
    args = load_config(cli_args.config)
    if cli_args.no_differential:
        args.no_differential = True

    # Forcer train=True pour le mode train
    args.train = True

    # Chemins
    data_dir = os.path.join(BASE_DIR, "data", "processed")
    out_dir = os.path.join(BASE_DIR, "output")
    models_dir = os.path.join(out_dir, "models")
    logs_dir = os.path.join(out_dir, "logs")
    evi_meta = os.path.join(data_dir, "release_evidences.json")
    patho_meta = os.path.join(data_dir, "release_conditions.json")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    args.evi_meta_path = evi_meta
    args.patho_meta_path = patho_meta

    # Device
    device = cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Environnements ────────────────────────────────────────────────────────
    print("\n[1/4] Initialisation de l'environnement d'entraînement...")
    train_path = os.path.join(data_dir, "train.csv")
    env = environment(args, train_path, train=True)
    print(f"  Train : {env.sample_size} patients, {env.symptom_size} symptômes, {env.diag_size} pathologies")

    print("[2/4] Initialisation de l'environnement de validation...")
    val_path = os.path.join(data_dir, "validate.csv")
    eval_env = environment(args, val_path, train=False)
    print(f"  Val   : {eval_env.sample_size} patients")

    # ── Agent ─────────────────────────────────────────────────────────────────
    print("[3/4] Création de l'agent AARLC...")
    agent = Policy_Gradient_pair_model(
        state_size=env.state_size,
        disease_size=env.diag_size,
        symptom_size=env.symptom_size,
        LR=args.lr,
        Gamma=args.gamma,
        device=device,
    )

    # Seuil adaptatif initial (Adaptive Alignment)
    threshold = args.threshold * np.ones(env.diag_size)
    lamb = args.lamb  # coefficient Polyak

    best_val_accuracy = 0.0
    patience_counter = 0
    log_history = []

    print("[4/4] Démarrage de l'entraînement AARLC...\n")
    print("=" * 60)

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(args.EPOCHS):
        env.reset()
        agent.train()

        num_batches = max(1, env.sample_size // args.batch_size)
        epoch_rewards = []
        epoch_correct = 0
        epoch_total = 0

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1:3d}/{args.EPOCHS}"):

            # Initialiser un batch de patients
            s, disease, diff_indices, diff_probas, disease_severity = env.initialize_state(args.batch_size)
            done = np.zeros(args.batch_size, dtype=bool)
            right_diagnosis = np.zeros(args.batch_size, dtype=bool)

            # Entropie initiale du classifier
            a_d_0, p_d_0 = agent.choose_diagnosis(s)
            ent_init = scipy_entropy(p_d_0, axis=1)
            ent = ent_init.copy()
            ent_init[ent_init < 1e-8] = 1.0

            # Buffers REINFORCE
            log_probs_buf = []
            entropy_buf = []
            rewards_buf = []
            states_buf = [s.copy()]

            # Rollout
            for step in range(args.MAXSTEP):
                a_p, log_prob, ent_policy = agent.choose_action_s(s)
                s, reward_s, done, right_diagnosis, diag, ent, a_d_ = env.step(
                    s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent
                )
                log_probs_buf.append(log_prob)
                entropy_buf.append(ent_policy)
                rewards_buf.append(reward_s.copy())
                states_buf.append(s.copy())

                # Mise à jour du seuil adaptatif (Polyak)
                if diag.any():
                    for j, pi in enumerate(a_d_[diag]):
                        threshold[pi] = lamb * threshold[pi] + (1 - lamb) * ent[diag][j]

                if np.all(done):
                    break

            # Calculer les retours
            returns_per_patient = []
            T = len(rewards_buf)
            for p_idx in range(args.batch_size):
                G = 0.0
                ep_returns = []
                for t in reversed(range(T)):
                    G = rewards_buf[t][p_idx] + args.gamma * G
                    ep_returns.insert(0, G)
                returns_per_patient.append(ep_returns)

            # Reformatter pour update_param_rl
            all_log_probs = [lp for lp in log_probs_buf]
            all_entropy = [e for e in entropy_buf]
            flat_returns = [[returns_per_patient[p][t] for p in range(args.batch_size)] for t in range(T)]

            # Update policy
            loss_rl = agent.update_param_rl(
                states=states_buf,
                actions=[],
                returns=flat_returns,
                entropy_list=all_entropy,
                log_probs_list=all_log_probs,
            )

            # Update classifier (sur tous les états du rollout)
            all_states = np.vstack(states_buf)
            target_diffs = np.tile(env.target_differential, (len(states_buf), 1))
            loss_c = agent.update_param_c(
                states=list(all_states),
                target_diffs=list(target_diffs),
            )

            # Métriques batch
            batch_rewards = np.array([rewards_buf[t] for t in range(T)]).sum(axis=0)
            epoch_rewards.extend(batch_rewards.tolist())
            epoch_correct += right_diagnosis.sum()
            epoch_total += args.batch_size

        # ── Évaluation validation ──────────────────────────────────────────────
        val_acc, val_steps = evaluate(
            eval_env, agent, args.eval_batch_size, threshold, args.MAXSTEP, device
        )
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        mean_reward = np.mean(epoch_rewards)

        log_entry = {
            "epoch": epoch + 1,
            "train_accuracy": round(float(train_acc), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_avg_steps": round(float(val_steps), 2),
            "mean_reward": round(float(mean_reward), 4),
            "loss_rl": round(float(loss_rl), 4),
            "loss_classifier": round(float(loss_c), 4),
        }
        log_history.append(log_entry)
        print(
            f"  Epoch {epoch+1:3d} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | "
            f"avg_steps={val_steps:.1f} | reward={mean_reward:.3f}"
        )

        # Early stopping + sauvegarde meilleur modèle
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            model_path = os.path.join(models_dir, "best_model.pkl")
            agent.save(model_path)
            # Sauvegarder le seuil adaptatif
            with open(os.path.join(models_dir, "threshold.pkl"), "wb") as f:
                pickle.dump(threshold, f)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping après {epoch+1} epochs (patience={args.patience})")
                break

    # ── Sauvegarde des logs ────────────────────────────────────────────────────
    logs_path = os.path.join(logs_dir, "training_log.json")
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f" ENTRAÎNEMENT TERMINÉ")
    print(f"   Best val accuracy : {best_val_accuracy:.4f}")
    print(f"   Modèle sauvegardé : output/models/best_model.pkl")
    print(f"   Logs              : output/logs/training_log.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
