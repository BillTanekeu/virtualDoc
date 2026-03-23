"""
env/sim_utils.py — Utilitaires de chargement de données
========================================================
Adapté de code/aarlc/ddxplus_code/sim_utils.py pour le projet maladies_fr.

# ADAPTATION: Chargement depuis data/processed/ au lieu des fichiers DDXPlus originaux
# ADAPTATION: Pas de race/ethnicity (données non disponibles)
# ADAPTATION: Support des bins discrétisés (gravite, duree_symptomes_jours)
"""

import ast
import json
import warnings
import os

import numpy as np
import pandas as pd


# ─── Encodage démographique ──────────────────────────────────────────────────

def encode_age(age: int) -> np.ndarray:
    """
    Encode l'âge en vecteur one-hot de 8 bins.
    Bins: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    """
    bins = [10, 20, 30, 40, 50, 60, 70, float("inf")]
    vec = np.zeros(len(bins), dtype=np.float32)
    for i, threshold in enumerate(bins):
        if age < threshold:
            vec[i] = 1.0
            return vec
    vec[-1] = 1.0
    return vec


def encode_sex(sex: str) -> np.ndarray:
    """
    Encode le sexe en vecteur one-hot de 2 valeurs.
    [M, F]
    """
    vec = np.zeros(2, dtype=np.float32)
    if sex == "M":
        vec[0] = 1.0
    else:
        vec[1] = 1.0
    return vec


# ADAPTATION: Race et ethnicity non disponibles dans ces données
def encode_race(race: str) -> np.ndarray:
    """Non utilisé. Retourne vecteur vide."""
    return np.zeros(0, dtype=np.float32)


def encode_ethnicity(ethnicity: str) -> np.ndarray:
    """Non utilisé. Retourne vecteur vide."""
    return np.zeros(0, dtype=np.float32)


def encode_geo(geo: str) -> np.ndarray:
    """Non utilisé. Retourne vecteur vide."""
    return np.zeros(0, dtype=np.float32)


# ─── Utilitaires de preprocessing ────────────────────────────────────────────

def preprocess_symptoms(symptoms) -> list:
    """
    Parse la colonne EVIDENCES (JSON list ou string legacy).
    """
    if pd.isnull(symptoms):
        return []
    s = str(symptoms).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []
    return []


def preprocess_differential(differential) -> list:
    """
    Parse la colonne DIFFERENTIAL_DIAGNOSIS.
    Format attendu: [[patho, proba], ...]
    Retourne: [[patho, proba], ...]
    """
    if pd.isnull(differential):
        return []
    s = str(differential).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []
    return []


def get_symptoms_with_multiple_answers(symptoms: list) -> list:
    """
    Identifie les symptômes catégoriels (présents via '_@_') qui apparaissent plusieurs fois.
    Ces symptômes ont plusieurs valeurs possibles → ils ont des réponses multiples.
    """
    roots = []
    for s in symptoms:
        idx = s.find("_@_")
        roots.append(s[:idx] if idx != -1 else s)
    
    count = {}
    result = []
    for r in roots:
        count[r] = count.get(r, 0) + 1
        if count[r] == 2:
            result.append(r)
    return result


def only_contain_derivated_symptoms(symptoms: list) -> bool:
    """Retourne True si TOUTES les évidences sont des valeurs '@'."""
    return all("_@_" in s for s in symptoms)


def stringify_differential(differential) -> str:
    """
    Convertit [[patho, proba], ...] en format legacy 'patho:sommeOR:score;...'
    pour compatibilité avec preprocess_differential de sim_utils original.
    """
    if not differential:
        return None
    is_str = isinstance(differential, str)
    if is_str and differential.startswith("["):
        differential = ast.literal_eval(differential)
    result = []
    for x in differential:
        patho, proba = x
        proba = float(proba)
        sommeOR = proba / (1 - proba) if proba < 1.0 else 100.0
        result.append(f"{patho}:{sommeOR}:1")
    return ";".join(result)


def convert_to_compatible_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit le format CSV DDXPlus release vers le format interne attendu par l'environnement.
    
    # ADAPTATION: Pas de race/ethnicity réelles → valeurs factices pour compatibilité API
    """
    df = df.rename(columns={
        "AGE": "AGE_BEGIN",
        "DIFFERENTIAL_DIAGNOSIS": "DIFFERNTIAL_DIAGNOSIS",
        "SEX": "GENDER",
        "EVIDENCES": "SYMPTOMS",
    })

    # Données factices pour compatibilité (non utilisées dans le calcul d'état)
    n = len(df)
    df["NUM_SYMPTOMS"] = [5] * n
    df["ETHNICITY"] = ["nonhispanic"] * n   # ADAPTATION: factice
    df["FOUND_GT_PATHOLOGY"] = [True] * n
    df["RACE"] = ["white"] * n              # ADAPTATION: factice

    # Convertir DIFFERENTIAL_DIAGNOSIS → format legacy string
    df["DIFFERNTIAL_DIAGNOSIS"] = df["DIFFERNTIAL_DIAGNOSIS"].apply(stringify_differential)

    if "INITIAL_EVIDENCE" in df.columns:
        df = df.rename(columns={"INITIAL_EVIDENCE": "INITIAL_SYMPTOM"})

    return df


def convert_to_compatible_json_format(data: dict) -> dict:
    """
    Convertit release_evidences.json / release_conditions.json vers format interne.
    Renomme les clés data_type→type-donnes et severity→urgence.
    """
    for x in data.keys():
        if "data_type" in data[x]:
            data[x]["type-donnes"] = data[x]["data_type"]
            data[x].pop("data_type")
        if "severity" in data[x]:
            data[x]["urgence"] = data[x]["severity"]
            data[x].pop("severity")
    return data


# ─── Chargement principal ─────────────────────────────────────────────────────

def load_csv(filepath: str, map_to_ids: bool = True):
    """
    Charge et prétraite le fichier CSV DDXPlus procesé pour le projet maladies_fr.

    # ADAPTATION: Chargement direct depuis data/processed/ (format DDXPlus déjà présent)
    # ADAPTATION: Pas de filtrage race/ethnicity

    Retourne le même tuple que load_csv original de sim_utils AARLC pour compatibilité.
    """
    # Gérer les fichiers zip si nécessaire
    if str(filepath).endswith(".zip"):
        df = pd.read_csv(filepath, compression="zip")
    else:
        df = pd.read_csv(filepath)

    # Convertir au format interne
    df = convert_to_compatible_format(df)

    # Parser les symptômes
    df["SYMPTOMS"] = df["SYMPTOMS"].apply(preprocess_symptoms)
    df["DIFFERNTIAL_DIAGNOSIS"] = df["DIFFERNTIAL_DIAGNOSIS"].apply(preprocess_differential)

    # Filtrer les lignes sans symptômes ou uniquement avec des dérivés
    df = df[df["SYMPTOMS"].apply(lambda x: len(x) > 0)]
    df = df[df["SYMPTOMS"].apply(lambda x: not only_contain_derivated_symptoms(x))]

    if len(df) == 0:
        return [pd.DataFrame()] + [{}] * 10

    # Construire les mappings id↔valeur
    unique_symptoms = _build_unique_mapping(df["SYMPTOMS"])
    unique_pathos = _build_unique_mapping_flat(df["PATHOLOGY"].tolist())
    unique_races = {"white": 0}     # ADAPTATION: valeur factice
    unique_ethnics = {"nonhispanic": 0}  # ADAPTATION: valeur factice
    unique_genders = _build_unique_mapping_flat(df["GENDER"].tolist())

    # Symptômes par pathologie
    patho_symptoms = {}
    for _, row in df.iterrows():
        patho_id = unique_pathos[row["PATHOLOGY"]]
        if patho_id not in patho_symptoms:
            patho_symptoms[patho_id] = set()
        for s in row["SYMPTOMS"]:
            root = s.split("_@_")[0]
            if root in unique_symptoms:
                patho_symptoms[patho_id].add(unique_symptoms[root])

    # Symptômes avec réponses multiples
    all_symptoms_flat = [s for row in df["SYMPTOMS"] for s in row]
    symptoms_with_multiple_answers = get_symptoms_with_multiple_answers(all_symptoms_flat)

    # Longueur max du diagnostic différentiel
    max_differential_len = df["DIFFERNTIAL_DIAGNOSIS"].apply(len).max()

    # Pathologies du différentiel
    unique_differential_pathos = _build_unique_mapping_flat(
        [p for row in df["DIFFERNTIAL_DIAGNOSIS"] for p, _ in row]
    )

    # Symptômes initiaux
    if "INITIAL_SYMPTOM" in df.columns:
        unique_init_symptoms = _build_unique_mapping_flat(
            df["INITIAL_SYMPTOM"].dropna().tolist()
        )
    else:
        unique_init_symptoms = {}

    return [
        df,
        unique_symptoms,
        unique_pathos,
        patho_symptoms,
        unique_races,
        unique_ethnics,
        unique_genders,
        symptoms_with_multiple_answers,
        max_differential_len,
        unique_differential_pathos,
        unique_init_symptoms,
    ]


def load_and_check_data(filepath: str, symptom_data: dict, patho_data: dict) -> dict:
    """
    Charge et valide les données JSON (evidences/conditions).
    Compatible avec l'API de sim_utils AARLC original.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ─── Helpers privés ──────────────────────────────────────────────────────────

def _build_unique_mapping(symptoms_series) -> dict:
    """
    Construit un mapping {symptom_str → unique_id} à partir des listes de symptômes.
    Les symptômes avec '_@_' sont ajoutés par leur chaîne complète ET leur racine.
    """
    all_vals = set()
    for sym_list in symptoms_series:
        for s in sym_list:
            all_vals.add(s)
    return {v: i for i, v in enumerate(sorted(all_vals))}


def _build_unique_mapping_flat(values: list) -> dict:
    """Construit un mapping {valeur → id} à partir d'une liste plate."""
    unique = sorted(set(str(v) for v in values if pd.notna(v)))
    return {v: i for i, v in enumerate(unique)}
