"""
Script 01 — Preprocessing et discrétisation
============================================
Ce script prépare les données brutes pour l'entraînement AARLC.

Fonctions principales :
1. Discrétisation des attributs catégoriels continus :
   - `gravite` (1–10) → 3 bins sémantiques
   - `duree_symptomes_jours` (1–30) → 4 bins sémantiques
2. Mise à jour des fichiers JSON (release_evidences, release_conditions)
3. Génération de data/processed/ et data/processed/meta.json

# ADAPTATION: Les données sources n'ont pas de tests physiques (version no_tests).
# ADAPTATION: Discrétisation ajoutée pour réduire la cardinalité de l'espace d'état RL.

Usage:
    python3 scripts/01_preprocess.py
"""

import csv
import json
import os
import ast

# ─── Chemins ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ─── Règles de discrétisation ────────────────────────────────────────────────

# ADAPTATION: Discrétisation de 'gravite' en 3 bins sémantiques (1–10 → legere/moderee/severe)
GRAVITE_BINS = {
    "legere":   list(range(1, 4)),    # 1, 2, 3
    "moderee":  list(range(4, 8)),    # 4, 5, 6, 7
    "severe":   list(range(8, 11)),   # 8, 9, 10
}

# ADAPTATION: Discrétisation de 'duree_symptomes_jours' en 4 bins sémantiques (1–30 → courte/semaine/longue/chronique)
DUREE_BINS = {
    "courte":    list(range(1, 8)),    # 1–7 jours
    "semaine":   list(range(8, 15)),   # 8–14 jours
    "longue":    list(range(15, 22)),  # 15–21 jours
    "chronique": list(range(22, 31)), # 22–30 jours
}

# Noms des attributs catégoriels à discrétiser
CATEGORICAL_ATTRS = {
    "gravite": GRAVITE_BINS,
    "duree_symptomes_jours": DUREE_BINS,
}


def discretize_value(attr_name: str, raw_value: str) -> str:
    """
    Convertit une valeur numérique en bin sémantique.
    
    Returns:
        Le label du bin (ex: "legere", "courte") ou la valeur originale si non reconnue.
    """
    bins = CATEGORICAL_ATTRS.get(attr_name)
    if bins is None:
        return raw_value
    try:
        int_val = int(raw_value)
    except (ValueError, TypeError):
        return raw_value

    for bin_label, bin_range in bins.items():
        if int_val in bin_range:
            return bin_label
    # Valeur hors plage → on garde en str
    return raw_value


def update_evidences_list(evidences_list: list) -> list:
    """
    Met à jour une liste d'évidences pour remplacer les valeurs catégorielles
    numériques par leurs bins discrétisés.
    
    Ex: ["gravite", "gravite_@_7", "fatigue"] → ["gravite", "gravite_@_moderee", "fatigue"]
    """
    updated = []
    for ev in evidences_list:
        replaced = False
        for attr_name in CATEGORICAL_ATTRS:
            prefix = f"{attr_name}_@_"
            if ev.startswith(prefix):
                raw_val = ev[len(prefix):]
                disc_val = discretize_value(attr_name, raw_val)
                updated.append(f"{prefix}{disc_val}")
                replaced = True
                break
        if not replaced:
            updated.append(ev)
    # Dédupliquer en préservant l'ordre
    seen = set()
    result = []
    for e in updated:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


def load_evidences_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_evidences_json(evidences: dict) -> dict:
    """
    Met à jour release_evidences.json pour que les attributs catégoriels 
    discrétisés aient des possible-values reflétant les bins.
    """
    updated = dict(evidences)
    for attr_name, bins in CATEGORICAL_ATTRS.items():
        if attr_name in updated:
            bin_labels = list(bins.keys())
            updated[attr_name]["possible-values"] = bin_labels
            # ADAPTATION: remplacement des value_meaning numériques par les labels de bins
            updated[attr_name]["value_meaning"] = {
                label: {"fr": label, "en": label} for label in bin_labels
            }
    return updated


def process_csv(input_path: str, output_path: str) -> None:
    """
    Lit un fichier CSV DDXPlus, applique la discrétisation sur les EVIDENCES
    et écrit le résultat dans output_path.
    """
    fieldnames = ["AGE", "DIFFERENTIAL_DIAGNOSIS", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"]
    rows_out = []

    with open(input_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            # Parser EVIDENCES (JSON list)
            try:
                ev_list = json.loads(row["EVIDENCES"])
            except (json.JSONDecodeError, KeyError):
                ev_list = []

            ev_list_disc = update_evidences_list(ev_list)

            # Discrétiser INITIAL_EVIDENCE si c'est une valeur catégorielle
            # (l'INITIAL_EVIDENCE est un symptôme racine, pas une valeur @, donc pas de changement)
            rows_out.append({
                "AGE": row["AGE"],
                "DIFFERENTIAL_DIAGNOSIS": row["DIFFERENTIAL_DIAGNOSIS"],
                "SEX": row["SEX"],
                "PATHOLOGY": row["PATHOLOGY"],
                "EVIDENCES": json.dumps(ev_list_disc, ensure_ascii=False),
                "INITIAL_EVIDENCE": row["INITIAL_EVIDENCE"],
            })

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"   {os.path.basename(input_path)} → {os.path.basename(output_path)} ({len(rows_out)} patients)")


def build_meta_json(conditions: dict, evidences: dict, output_path: str) -> None:
    """
    Génère meta.json avec les mappings index↔nom attendus par sim_utils.py AARLC.
    
    Structure :
    {
      "conditions": {"Asthme": 0, ...},
      "evidences":  {"gravite": 0, ...},
      "evidences_list": ["gravite", "duree_symptomes_jours", ...],
      "conditions_list": ["Asthme", "Meningite", ...]
    }
    """
    conditions_list = sorted(conditions.keys())
    evidences_list = sorted(evidences.keys())

    meta = {
        "conditions": {name: idx for idx, name in enumerate(conditions_list)},
        "evidences":  {name: idx for idx, name in enumerate(evidences_list)},
        "conditions_list": conditions_list,
        "evidences_list":  evidences_list,
        "antecedents": [
            name for name, ev in evidences.items()
            if ev.get("is_antecedent", False)
        ],
        "discretization_bins": {
            attr: list(bins.keys()) for attr, bins in CATEGORICAL_ATTRS.items()
        },
        "stats": {
            "n_conditions":  len(conditions_list),
            "n_evidences":   len(evidences_list),
            "n_antecedents": sum(
                1 for ev in evidences.values() if ev.get("is_antecedent", False)
            ),
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"   meta.json généré ({len(conditions_list)} conditions, {len(evidences_list)} évidences)")


def main():
    print("=" * 60)
    print("PREPROCESSING — Discrétisation et préparation des données")
    print("=" * 60)

    os.makedirs(PROC_DIR, exist_ok=True)

    # ── 1. Charger et mettre à jour release_evidences.json ──────────────────
    print("\n[1/4] Mise à jour de release_evidences.json...")
    ev_path = os.path.join(RAW_DIR, "release_evidences.json")
    evidences = load_evidences_json(ev_path)
    evidences_updated = update_evidences_json(evidences)

    ev_proc_path = os.path.join(PROC_DIR, "release_evidences.json")
    with open(ev_proc_path, "w", encoding="utf-8") as f:
        json.dump(evidences_updated, f, indent=4, ensure_ascii=False)
    print(f"   release_evidences.json mis à jour")
    for attr, bins in CATEGORICAL_BINS_SUMMARY(evidences_updated).items():
        print(f"     {attr}: {bins}")

    # ── 2. Copier release_conditions.json ───────────────────────────────────
    print("\n[2/4] Copie de release_conditions.json...")
    cond_path = os.path.join(RAW_DIR, "release_conditions.json")
    with open(cond_path, "r", encoding="utf-8") as f:
        conditions = json.load(f)
    cond_proc_path = os.path.join(PROC_DIR, "release_conditions.json")
    with open(cond_proc_path, "w", encoding="utf-8") as f:
        json.dump(conditions, f, indent=4, ensure_ascii=False)
    print(f"  release_conditions.json ({len(conditions)} conditions)")

    # ── 3. Traiter les splits CSV ────────────────────────────────────────────
    print("\n[3/4] Discrétisation des splits CSV...")
    for split in ["train", "validate", "test"]:
        inp = os.path.join(RAW_DIR, f"{split}.csv")
        out = os.path.join(PROC_DIR, f"{split}.csv")
        if os.path.exists(inp):
            process_csv(inp, out)
        else:
            print(f"  ⚠️  {split}.csv introuvable dans data/raw/")

    # ── 4. Générer meta.json ─────────────────────────────────────────────────
    print("\n[4/4] Génération de meta.json...")
    meta_path = os.path.join(PROC_DIR, "meta.json")
    build_meta_json(conditions, evidences_updated, meta_path)

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINÉ")
    print(f"  Données discrétisées dans : data/processed/")
    print(f"  Bins gravite      : {list(GRAVITE_BINS.keys())}")
    print(f"  Bins duree_jours  : {list(DUREE_BINS.keys())}")
    print("=" * 60)


def CATEGORICAL_BINS_SUMMARY(evidences: dict) -> dict:
    """Helper pour afficher les bins des attributs catégoriels."""
    result = {}
    for attr in CATEGORICAL_ATTRS:
        if attr in evidences:
            result[attr] = evidences[attr].get("possible-values", [])
    return result


if __name__ == "__main__":
    main()
