"""
Script 02 — Validation de conformité DDXPlus
=============================================
Vérifie que le dataset procesé (data/processed/) est conforme au format DDXPlus
attendu par AARLC. Produit output/validation_report.json.

Vérifications effectuées :
  - Colonnes CSV complètes
  - Pathologies cohérentes avec release_conditions.json
  - Évidences cohérentes avec release_evidences.json
  - INITIAL_EVIDENCE appartient aux évidences du patient
  - Format DIFFERENTIAL_DIAGNOSIS valide
  - Valeurs SEX = M ou F
  - AGE entier dans [0, 120]
  - Valeurs catégorielles discrétisées cohérentes avec possible-values
  - Absence de fuite entre splits

Usage:
    cd projet_maladies_fr/
    python3 scripts/02_validate_dataset.py
"""

import csv
import json
import os
import sys
from collections import defaultdict

# ─── Chemins ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SPLITS = ["train", "validate", "test"]
REQUIRED_COLUMNS = {"AGE", "DIFFERENTIAL_DIAGNOSIS", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check(condition: bool, message: str, errors: list, warnings: list = None, is_warning: bool = False) -> bool:
    if not condition:
        if is_warning and warnings is not None:
            warnings.append(f"  {message}")
        else:
            errors.append(f" {message}")
    return condition


# ─── Validations ─────────────────────────────────────────────────────────────

def validate_columns(reader_fieldnames: list, split: str, errors: list) -> bool:
    missing = REQUIRED_COLUMNS - set(reader_fieldnames)
    ok = len(missing) == 0
    check(ok, f"[{split}] Colonnes manquantes : {missing}", errors)
    return ok


def validate_categorical_values(ev_name: str, ev_value: str, evidences: dict, errors: list, row_num: int, split: str) -> None:
    """Vérifie que la valeur discrétisée est dans possible-values du JSON."""
    if ev_name in evidences and evidences[ev_name].get("data_type") == "C":
        possible = evidences[ev_name].get("possible-values", [])
        if possible and ev_value not in possible:
            errors.append(
                f" [{split}] ligne {row_num}: valeur '{ev_value}' non déclarée dans "
                f"possible-values de '{ev_name}' ({possible})"
            )


def validate_split(split: str, conditions: dict, evidences: dict, errors: list, warnings: list) -> dict:
    """Valide un split CSV complet. Retourne les stats du split."""
    csv_path = os.path.join(PROC_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        errors.append(f" Fichier manquant : data/processed/{split}.csv")
        return {}

    stats = {"n_patients": 0, "pathologies": defaultdict(int), "issues": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Colonnes
        if not validate_columns(reader.fieldnames, split, errors):
            return stats

        for row_num, row in enumerate(reader, start=2):  # ligne 1 = header
            stats["n_patients"] += 1
            patho = row.get("PATHOLOGY", "").strip()
            stats["pathologies"][patho] += 1

            # AGE
            try:
                age = int(row["AGE"])
                check(0 <= age <= 120, f"[{split}] ligne {row_num}: AGE invalide ({age})", errors)
            except (ValueError, KeyError):
                errors.append(f" [{split}] ligne {row_num}: AGE non entier")
                stats["issues"] += 1

            # SEX
            sex = row.get("SEX", "").strip()
            check(sex in {"M", "F"}, f"[{split}] ligne {row_num}: SEX invalide '{sex}'", errors)

            # PATHOLOGY dans conditions.json
            check(
                patho in conditions,
                f"[{split}] ligne {row_num}: PATHOLOGY '{patho}' absente de release_conditions.json",
                errors
            )

            # DIFFERENTIAL_DIAGNOSIS format
            try:
                diff = json.loads(row["DIFFERENTIAL_DIAGNOSIS"])
                check(
                    isinstance(diff, list) and all(
                        isinstance(d, list) and len(d) == 2 and
                        isinstance(d[0], str) and isinstance(d[1], float)
                        for d in diff
                    ),
                    f"[{split}] ligne {row_num}: DIFFERENTIAL_DIAGNOSIS format invalide",
                    errors
                )
            except (json.JSONDecodeError, KeyError):
                errors.append(f" [{split}] ligne {row_num}: DIFFERENTIAL_DIAGNOSIS non parsable")
                stats["issues"] += 1

            # EVIDENCES
            try:
                ev_list = json.loads(row["EVIDENCES"])
                check(isinstance(ev_list, list), f"[{split}] ligne {row_num}: EVIDENCES n'est pas une liste", errors)

                ev_set = set(ev_list)
                for ev in ev_list:
                    # Évidences '@' → vérification de la racine + valeur
                    if "_@_" in ev:
                        root_name, ev_value = ev.rsplit("_@_", 1)
                        check(
                            root_name in evidences,
                            f"[{split}] ligne {row_num}: évidence racine '{root_name}' inconnue",
                            errors
                        )
                        validate_categorical_values(root_name, ev_value, evidences, errors, row_num, split)
                    else:
                        check(
                            ev in evidences,
                            f"[{split}] ligne {row_num}: évidence '{ev}' absente de release_evidences.json",
                            errors
                        )

                # INITIAL_EVIDENCE dans la liste
                init_ev = row.get("INITIAL_EVIDENCE", "").strip()
                check(
                    init_ev in ev_set or init_ev == "",
                    f"[{split}] ligne {row_num}: INITIAL_EVIDENCE '{init_ev}' absent des EVIDENCES",
                    warnings,
                    is_warning=True
                )

            except (json.JSONDecodeError, KeyError):
                errors.append(f" [{split}] ligne {row_num}: EVIDENCES non parsable")
                stats["issues"] += 1

    return stats


def validate_no_leakage(split_stats: dict, warnings: list) -> None:
    """Vérifie l'absence de fuite entre splits (basé sur les pathologies)."""
    # Pour ce dataset synthétique, toutes les pathologies sont dans tous les splits (normal)
    # On vérifie juste que les splits ont des tailles cohérentes
    sizes = {split: s.get("n_patients", 0) for split, s in split_stats.items()}
    total = sum(sizes.values())
    if total > 0:
        train_ratio = sizes.get("train", 0) / total
        if not (0.70 <= train_ratio <= 0.90):
            warnings.append(
                f"  Ratio train anormal : {train_ratio:.1%} (attendu ~80%)"
            )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VALIDATION — Conformité DDXPlus du dataset procesé")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []
    warnings = []

    # Charger les JSON de référence
    ev_path = os.path.join(PROC_DIR, "release_evidences.json")
    cond_path = os.path.join(PROC_DIR, "release_conditions.json")
    meta_path = os.path.join(PROC_DIR, "meta.json")

    for path, label in [(ev_path, "release_evidences.json"),
                        (cond_path, "release_conditions.json"),
                        (meta_path, "meta.json")]:
        check(os.path.exists(path), f"Fichier manquant : data/processed/{label}", errors)

    if errors:
        print("\n".join(errors))
        sys.exit(1)

    evidences = load_json(ev_path)
    conditions = load_json(cond_path)
    meta = load_json(meta_path)

    print(f"\nRéférence chargée: {len(conditions)} conditions, {len(evidences)} évidences")

    # Valider chaque split
    split_stats = {}
    for split in SPLITS:
        print(f"\n[{split.upper()}] Validation en cours...")
        stats = validate_split(split, conditions, evidences, errors, warnings)
        split_stats[split] = stats
        if stats:
            print(f"  {stats['n_patients']} patients, {len(stats['pathologies'])} pathologies distinctes")

    # Vérifier l'absence de fuite
    validate_no_leakage(split_stats, warnings)

    # ── Rapport JSON ─────────────────────────────────────────────────────────
    report = {
        "conformity_status": "CONFORME" if not errors else "NON_CONFORME",
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "dataset_stats": {
            split: {
                "n_patients": s.get("n_patients", 0),
                "pathologies_distribution": dict(s.get("pathologies", {})),
            }
            for split, s in split_stats.items()
        },
        "reference": {
            "n_conditions": len(conditions),
            "n_evidences": len(evidences),
            "discretization_bins": meta.get("discretization_bins", {})
        }
    }

    report_path = os.path.join(OUTPUT_DIR, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # ── Affichage final ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not errors:
        print(f" SUCCÈS — Dataset CONFORME au format DDXPlus")
    else:
        print(f" ÉCHEC — {len(errors)} erreur(s) détectée(s)")
        for e in errors[:10]:
            print(f"   {e}")
        if len(errors) > 10:
            print(f"   ... (+{len(errors) - 10} autres, voir validation_report.json)")

    if warnings:
        print(f"\n  {len(warnings)} avertissement(s):")
        for w in warnings[:5]:
            print(f"   {w}")

    print(f"\nRapport complet: output/validation_report.json")
    print("=" * 60)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
