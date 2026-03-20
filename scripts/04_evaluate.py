"""
Script 04 — Évaluation du modèle entraîné
==========================================
Charge le meilleur modèle (output/models/best_model.pkl) et l'évalue
sur le test set. Produit output/evaluation_results.json.

Métriques calculées :
  - Accuracy@1 (top-1 diagnostic)
  - Avg Steps (nombre moyen d'interrogations par patient)
  - DDR / DDP / DDF1 (Differential Diagnosis Recall/Precision/F1)

Usage:
    cd projet_maladies_fr/
    python3 scripts/04_evaluate.py
"""

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


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return argparse.Namespace(**{
        k: v for k, v in cfg.items()
        if k not in ("description", "comments")
    })


def compute_differential_metrics(pred_indices, true_indices, n_pathos):
    """
    Calcule DDR/DDP/DDF1 pour un batch.
    
    DDR (Recall)   : proportion des vrais diagnostics retrouvés
    DDP (Precision): proportion des diagnostics prédits corrects
    DDF1 : F1 de (DDR, DDP)
    """
    tp = np.sum(pred_indices == true_indices)
    ddr = tp / len(true_indices) if len(true_indices) > 0 else 0.0
    ddp = tp / len(pred_indices) if len(pred_indices) > 0 else 0.0
    ddf1 = 2 * ddr * ddp / (ddr + ddp) if (ddr + ddp) > 0 else 0.0
    return ddr, ddp, ddf1


def evaluate_on_test(env, agent, threshold, max_step, device):
    """Évaluation complète sur le test set."""
    agent.eval()
    env.reset()

    batch_size = min(50, env.sample_size)
    total = env.sample_size
    num_batches = max(1, total // batch_size)

    all_correct = []
    all_steps = []
    all_preds = []
    all_truths = []

    with torch.no_grad():
        for b in range(num_batches):
            s, disease, diff_indices, diff_probas, _ = env.initialize_state(batch_size)
            done = np.zeros(batch_size, dtype=bool)
            right_diagnosis = np.zeros(batch_size, dtype=bool)
            steps_taken = np.zeros(batch_size, dtype=int)

            a_d_0, p_d_0 = agent.choose_diagnosis(s)
            ent_init = scipy_entropy(p_d_0, axis=1)
            ent = ent_init.copy()
            ent_init[ent_init < 1e-8] = 1.0

            for step in range(max_step):
                a_p, _, _ = agent.choose_action_s(s)
                s, _, done, right_diagnosis, diag, ent, a_d_ = env.step(
                    s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent
                )
                steps_taken[~done & ~diag] += 1
                if np.all(done):
                    break

            # Diagnostic final pour les patients non encore stoppés
            if not np.all(done):
                a_d_final, _ = agent.choose_diagnosis(s)
                final_correct = (a_d_final == np.array(env.disease)) & (~done)
                right_diagnosis |= final_correct
                all_preds.extend(a_d_final[~done].tolist())
            else:
                all_preds.extend(a_d_[done].tolist())

            all_correct.extend(right_diagnosis.tolist())
            all_steps.extend(steps_taken.tolist())
            all_truths.extend(env.disease.tolist())

    accuracy = np.mean(all_correct)
    avg_steps = np.mean(all_steps)
    ddr, ddp, ddf1 = compute_differential_metrics(
        np.array(all_preds), np.array(all_truths), env.num_pathos
    )

    return {
        "accuracy_at_1": round(float(accuracy), 4),
        "avg_steps": round(float(avg_steps), 2),
        "n_patients_evaluated": len(all_correct),
        "DDR": round(float(ddr), 4),
        "DDP": round(float(ddp), 4),
        "DDF1": round(float(ddf1), 4),
    }


def main():
    print("=" * 60)
    print("ÉVALUATION — Test Set (AARLC Maladies FR)")
    print("=" * 60)

    config_path = os.path.join(BASE_DIR, "configs", "config_maladies_fr.json")
    args = load_config(config_path)
    args.train = False

    data_dir = os.path.join(BASE_DIR, "data", "processed")
    models_dir = os.path.join(BASE_DIR, "output", "models")
    out_dir = os.path.join(BASE_DIR, "output")
    evi_meta = os.path.join(data_dir, "release_evidences.json")
    patho_meta = os.path.join(data_dir, "release_conditions.json")
    args.evi_meta_path = evi_meta
    args.patho_meta_path = patho_meta

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Charger le modèle ─────────────────────────────────────────────────────
    model_path = os.path.join(models_dir, "best_model.pkl")
    threshold_path = os.path.join(models_dir, "threshold.pkl")

    if not os.path.exists(model_path):
        print(f" Modèle introuvable : {model_path}")
        print("   Executez d'abord : python3 scripts/03_train.py")
        sys.exit(1)

    print(f"\n[1/3] Chargement de l'environnement test...")
    test_path = os.path.join(data_dir, "test.csv")
    test_env = environment(args, test_path, train=False)
    print(f"  Test : {test_env.sample_size} patients")

    print(f"\n[2/3] Chargement du modèle...")
    agent = Policy_Gradient_pair_model(
        state_size=test_env.state_size,
        disease_size=test_env.diag_size,
        symptom_size=test_env.symptom_size,
        device=device,
    )
    agent.load(model_path)

    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
        print(f"  Seuil adaptatif chargé.")
    else:
        threshold = args.threshold * np.ones(test_env.diag_size)
        print(f"  Seuil adaptatif par défaut : {args.threshold}")

    print(f"\n[3/3] Évaluation sur le test set...")
    results = evaluate_on_test(test_env, agent, threshold, args.MAXSTEP, device)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    report = {
        "model_path": model_path,
        "dataset": "data/processed/test.csv",
        "metrics": results,
        "config": {
            "MAXSTEP": args.MAXSTEP,
            "no_differential": getattr(args, "no_differential", False),
        }
    }
    results_path = os.path.join(out_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # ── Affichage ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" RÉSULTATS SUR LE TEST SET")
    print(f"   Accuracy@1   : {results['accuracy_at_1']:.1%}")
    print(f"   Avg Steps    : {results['avg_steps']:.1f} / {args.MAXSTEP}")
    print(f"   DDR          : {results['DDR']:.4f}")
    print(f"   DDP          : {results['DDP']:.4f}")
    print(f"   DDF1         : {results['DDF1']:.4f}")
    print(f"   N patients   : {results['n_patients_evaluated']}")
    print(f"\n   Rapport      : output/evaluation_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
