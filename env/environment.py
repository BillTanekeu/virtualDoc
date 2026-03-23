"""
env/environment.py — Environnement RL pour le projet maladies_fr
================================================================
Adapté de code/aarlc/ddxplus_code/env.py pour le projet maladies_fr.

# ADAPTATION: Import depuis env.sim_utils au lieu de sim_utils AARLC original
# ADAPTATION: Race/Ethnicity désactivées (non pertinent pour ces données)
# ADAPTATION: Dimensionnement pour 59 évidences au lieu de 223
# ADAPTATION: Chargement depuis data/processed/ 

Architecture fidèle à AARLC :
- État : [demo_features | symptômes (ternaire: 0/+1/-1)]
- Action : index de symptôme à interroger (0..N_sym-1)
- Reward : r_sym + ν·Δent + r_diag
"""

import random
import copy
import os
import json

import numpy as np
from scipy.stats import entropy

from sim_utils import (
    encode_age,
    encode_sex,
    encode_race,
    encode_ethnicity,
    load_csv,
    load_and_check_data,
    convert_to_compatible_json_format,
)

# ─── Valeurs ternaires de l'état ─────────────────────────────────────────────
NONE_VAL = 0   # Symptôme pas encore interrogé
PRES_VAL = 1   # Symptôme présent
ABS_VAL = -1   # Symptôme absent


class environment:
    """
    Environnement RL de simulation patient pour le projet maladies_fr.
    Fidèle à l'architecture de environment() dans AARLC/ddxplus_code/env.py.
    """

    def __init__(self, args, patient_filepath, train=None):
        """
        Initialise l'environnement.

        Parameters
        ----------
        args : namespace
            Arguments du programme principal (issus de config ou argparse).
        patient_filepath : str
            Chemin vers le CSV (data/processed/train.csv, etc.)
        train : bool, optional
            Mode entraînement (shuffle) ou évaluation.
        """
        self.args = args
        self.filepath = patient_filepath
        self.use_initial_symptom_flag = not getattr(args, "no_initial_evidence", False)
        self.max_turns = args.MAXSTEP
        self.action_type = 0  # Actions entières (index de symptôme)
        self.include_turns_in_state = getattr(args, "include_turns_in_state", False)
        # ADAPTATION: Race/Ethnicity désactivées
        self.include_race_in_state = False
        self.include_ethnicity_in_state = False
        self.use_differential_diagnosis = not getattr(args, "no_differential", False)
        self.train = getattr(args, "train", True) if train is None else train

        # ── Chargement des données ───────────────────────────────────────────
        [
            rb,
            self.unique_symptoms,
            self.unique_pathos,
            patho_symptoms,
            self.unique_races,
            self.unique_ethnics,
            self.unique_genders,
            self.symptoms_with_multiple_answers,
            self.max_differential_len,
            self.unique_differential_pathos,
            self.unique_init_symptoms,
        ] = load_csv(patient_filepath)

        if len(rb) == 0:
            raise ValueError("Le fichier ne contient aucun patient valide.")

        # Conversion patho_symptoms : id → noms
        inv_pathos = {v: k for k, v in self.unique_pathos.items()}
        inv_symptoms = {v: k for k, v in self.unique_symptoms.items()}
        patho_symptoms = {
            inv_pathos[a]: set(
                [inv_symptoms[b] for b in patho_symptoms.get(a, [])]
            )
            for a in patho_symptoms.keys()
        }

        # ── Chargement des métadonnées symptômes/pathologies ─────────────────
        self._load_and_check_symptoms_with_pathos(
            getattr(args, "evi_meta_path", None),
            self.unique_symptoms,
            self.symptoms_with_multiple_answers,
            getattr(args, "patho_meta_path", None),
            self.unique_pathos,
            patho_symptoms,
        )

        # ── Dimensions démographiques ─────────────────────────────────────────
        # ADAPTATION: 8 (age) + 2 (sex) = 10, pas de race/ethnicity
        self.num_age_values = 8
        self.num_sex_values = 2
        self.num_race_values = 0      # ADAPTATION: désactivé
        self.num_ethnic_values = 0    # ADAPTATION: désactivé
        self.num_demo_features = self.num_age_values + self.num_sex_values

        low_demo_values = [0] * self.num_demo_features
        high_demo_values = [1] * self.num_demo_features

        if self.include_turns_in_state:
            low_demo_values = [0] + low_demo_values
            high_demo_values = [1] + high_demo_values
            self.num_demo_features += 1

        # ── Espaces d'action et d'observation ─────────────────────────────────
        self.num_symptoms = len(self.symptom_index_2_key)
        self.num_pathos = len(self.pathology_index_2_key)
        if not (self.num_symptoms > 0 and self.num_pathos > 0):
            raise ValueError("Nombre de symptômes ou de pathologies nul.")

        self._define_action_and_observation_spaces(
            self.num_symptoms,
            self.num_pathos,
            self.num_demo_features,
            low_demo_values,
            high_demo_values,
            np.float32,
        )

        self.turns = 0
        self.context_size = (
            self.num_age_values
            + self.num_sex_values
            + self.num_race_values
            + self.num_ethnic_values
            + (1 if self.include_turns_in_state else 0)
        )
        self.sample_size = len(rb)
        self.idx = 0
        self.indexes = np.arange(self.sample_size)
        self.diag_size = self.num_pathos
        self.symptom_size = self.num_symptoms
        self.state_size = self.num_features - len(high_demo_values) + self.context_size

        # Matrices de coût / gain par symptôme
        self.cost = np.ones(self.symptom_size)
        self.earn = np.ones(self.symptom_size)
        self.action_mask = np.zeros((self.num_symptoms, self.state_size))
        self.symptom_mask = np.zeros((1, self.num_symptoms))
        self.atcd_mask = np.zeros((1, self.num_symptoms))
        for symptom_index in range(self.num_symptoms):
            start_idx = self.symptom_to_obs_mapping[symptom_index][0]
            end_idx = self.symptom_to_obs_mapping[symptom_index][1]
            self.action_mask[symptom_index, start_idx:end_idx] = 1
            is_atcd = self.symptom_data[self.symptom_index_2_key[symptom_index]].get("is_antecedent", False)
            if is_atcd:
                self.atcd_mask[0, symptom_index] = 1
            else:
                self.symptom_mask[0, symptom_index] = 1

        self.severity_mask = np.zeros((self.num_pathos,))
        severity_threshold = 3
        for pi in range(self.num_pathos):
            urgence = self.pathology_data[self.pathology_index_2_key[pi]].get("urgence", severity_threshold)
            if urgence < severity_threshold:
                self.severity_mask[pi] = 1

        self._put_patients_data_in_cache(rb)

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        self.idx = 0
        self.turns = 0
        if self.train:
            np.random.shuffle(self.indexes)

    # ─── Initialisation d'un batch ────────────────────────────────────────────

    def initialize_state(self, batch_size):
        self.batch_size = batch_size
        self.batch_index = self.indexes[self.idx: self.idx + batch_size]
        self.idx += batch_size
        self.disease = []
        self.differential_indices = []
        self.differential_probas = []
        self.disease_severity = []
        self.pos_sym = []
        self.acquired_sym = []

        i = 0
        init_state = np.ones((batch_size, self.state_size), dtype=self.obs_dtype) * NONE_VAL
        self.target_state = np.ones((batch_size, self.state_size), dtype=self.obs_dtype) * ABS_VAL
        self.all_state = np.zeros((batch_size, self.symptom_size))
        self.target_differential = np.zeros((batch_size, self.diag_size))
        self.inquired_symptoms = np.zeros((batch_size, self.symptom_size))

        if self.include_turns_in_state:
            init_state[:, 0] = 0
            self.target_state[:, 0] = 0

        for item in self.batch_index:
            cp = self.cached_patients[item]
            age = cp["age"]
            race = cp["race"]
            sex = cp["sex"]
            ethnic = cp["ethnic"]
            initial_symptom = cp["initial_symptom"]
            pathology_index = cp["pathology_index"]
            pathology_severity = cp["pathology_severity"]
            differential_indices = copy.deepcopy(cp["differential_indices"])
            differential_probas = copy.deepcopy(cp["differential_probas"])
            binary_symptoms = cp["bin_sym"]
            present_symptoms = cp["pres_sym"]
            self.target_state[i, :] = cp["tgt_state"][:]

            init_state[i, :] = self._init_demo_features(init_state[i, :], age, sex, race, ethnic)
            self.disease.append(pathology_index)
            self.disease_severity.append(pathology_severity)
            self.differential_indices.append(differential_indices)
            self.differential_probas.append(differential_probas)

            binary_symptoms = (
                [initial_symptom]
                if initial_symptom is not None and self.use_initial_symptom_flag
                else binary_symptoms
            )
            if binary_symptoms is None or len(binary_symptoms) == 0:
                first_symptom = random.choice(list(range(self.num_symptoms)))
            else:
                first_symptom = random.choice(binary_symptoms)

            frame_index = self.symptom_to_obs_mapping[first_symptom][0]
            init_state[i, frame_index] = PRES_VAL
            self.all_state[i, present_symptoms] = 1
            self.inquired_symptoms[i, first_symptom] = 1

            if differential_indices is not None and differential_probas is not None:
                valid = differential_indices[differential_indices != -1]
                self.target_differential[i, valid] = differential_probas[differential_indices != -1]
            else:
                self.target_differential[i, pathology_index] = 1.0

            i += 1

        self.disease = np.array(self.disease)
        self.disease_severity = np.array(self.disease_severity)
        if len(self.differential_indices) > 0 and self.differential_indices[0] is None:
            self.differential_indices = None
            self.differential_probas = None
        else:
            self.differential_indices = np.array(self.differential_indices)
            self.differential_probas = np.array(self.differential_probas)

        return init_state, self.disease, self.differential_indices, self.differential_probas, self.disease_severity

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent):
        """
        Effectue un pas de l'environnement.
        
        Returns: (s_, reward_s, done, right_diagnosis, diag, ent_, a_d_)
        """
        s_ = copy.deepcopy(s)
        ent_ = copy.deepcopy(ent)
        s_[~done] = (
            (1 - self.action_mask[a_p[~done]]) * s_[~done]
            + self.action_mask[a_p[~done]] * self.target_state[~done]
        )
        self.turns += 1
        if self.include_turns_in_state:
            s_[~done, 0] = self.turns / self.max_turns

        a_d_, p_d_ = agent.choose_diagnosis(s_)
        ent_[~done] = entropy(p_d_[~done], axis=1)
        ent_ratio = (ent - ent_) / ent_init

        diag = (ent_ < threshold[a_d_]) & (~done)
        right_diag = (a_d_ == np.array(self.disease)) & diag

        mu = getattr(self.args, "mu", 1.0)
        nu = getattr(self.args, "nu", 2.5)

        reward_s = mu * self.reward_func(s[:, self.context_size:], s_[:, self.context_size:], diag, a_p)
        reward_s[ent_ratio > 0] += nu * ent_ratio[ent_ratio > 0]
        reward_s[diag] -= mu * 1
        reward_s[right_diag] += mu * 2
        reward_s[done] = 0

        self.inquired_symptoms[~done, a_p[~done]] = 1
        done = done | diag
        right_diagnosis = right_diagnosis | right_diag

        return s_, reward_s, done, right_diagnosis, diag, ent_, a_d_

    def reward_func(self, s, s_, diag, a_p):
        reward = -self.cost[a_p]
        already_inquired = self.inquired_symptoms[range(len(a_p)), a_p]
        unrepeated = 1 - already_inquired
        reward += unrepeated * self.cost[a_p] * 0.7
        positive = self.all_state[range(len(a_p)), a_p]
        reward += positive * unrepeated * self.earn[a_p]
        return reward

    # ─── Helpers internes ────────────────────────────────────────────────────

    def _init_demo_features(self, state, age, sex, race, ethnic):
        """Encode les features démographiques dans l'état."""
        offset = 1 if self.include_turns_in_state else 0
        age_vec = encode_age(age)
        sex_vec = encode_sex(sex)
        state[offset: offset + self.num_age_values] = age_vec
        state[offset + self.num_age_values: offset + self.num_age_values + self.num_sex_values] = sex_vec
        # ADAPTATION: race et ethnicity non encodées
        return state

    def _load_and_check_symptoms_with_pathos(
        self, evi_meta_path, unique_symptoms, symptoms_mult, patho_meta_path, unique_pathos, patho_symptoms
    ):
        """Charge et valide les métadonnées symptômes et pathologies."""
        if evi_meta_path and os.path.exists(evi_meta_path):
            with open(evi_meta_path, "r", encoding="utf-8") as f:
                symptom_data = json.load(f)
            symptom_data = convert_to_compatible_json_format(symptom_data)
        else:
            # Génération automatique depuis unique_symptoms
            symptom_data = {
                name: {"name": name, "type-donnes": "B", "is_antecedent": False}
                for name in unique_symptoms
            }

        if patho_meta_path and os.path.exists(patho_meta_path):
            with open(patho_meta_path, "r", encoding="utf-8") as f:
                pathology_data = json.load(f)
            pathology_data = convert_to_compatible_json_format(pathology_data)
        else:
            pathology_data = {
                name: {"condition_name": name, "urgence": 1}
                for name in unique_pathos
            }

        # Construire les mappings index↔nom
        all_symptom_names = sorted(set(
            s.split("_@_")[0] for s in unique_symptoms
        ))
        self.symptom_index_2_key = {i: name for i, name in enumerate(sorted(symptom_data.keys()))}
        self.symptom_name_2_index = {name: i for i, name in self.symptom_index_2_key.items()}
        self.all_symptom_names = list(self.symptom_index_2_key.values())
        self.symptom_data = symptom_data

        self.pathology_index_2_key = {i: name for i, name in enumerate(sorted(pathology_data.keys()))}
        self.pathology_name_2_index = {name: i for i, name in self.pathology_index_2_key.items()}
        self.pathology_data = pathology_data
        self.pathology_severity_data = np.array([
            pathology_data[self.pathology_index_2_key[i]].get("urgence", 1)
            for i in range(len(self.pathology_index_2_key))
        ])

    def _define_action_and_observation_spaces(
        self, num_symptoms, num_pathos, num_demo_features,
        low_demo_values, high_demo_values, obs_dtype
    ):
        """Définit les espaces d'action et d'observation."""
        self.num_actions = [num_symptoms]  # Action = index de symptôme
        symp_low_val, symp_high_val = [], []
        symptom_to_obs_mapping = {}
        symptom_possible_val_mapping = {}
        symptom_data_types = {}
        symptom_default_value_mapping = {}
        categorical_integer_symptoms = set()

        for idx in range(len(self.symptom_index_2_key)):
            key = self.symptom_index_2_key[idx]
            data_type = self.symptom_data[key].get("type-donnes", "B")
            possible_values = self.symptom_data[key].get("possible-values", [])
            default_value = self.symptom_data[key].get("default_value", None)
            start_obs_idx = len(symp_low_val) + num_demo_features
            symptom_data_types[idx] = data_type
            num_elts = len(possible_values)

            if data_type == "B":
                symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
            elif data_type == "C":
                assert num_elts > 0, f"Symptôme catégoriel '{key}' sans possible-values"
                if isinstance(possible_values[0], str):
                    for _ in range(num_elts):
                        symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                        symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                    symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + num_elts]
                else:
                    categorical_integer_symptoms.add(idx)
                    symp_low_val.append(min(NONE_VAL, PRES_VAL))
                    symp_high_val.append(max(NONE_VAL, PRES_VAL))
                    symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
                symptom_possible_val_mapping[idx] = {a: i for i, a in enumerate(possible_values)}
                symptom_default_value_mapping[idx] = default_value
            elif data_type == "M":
                assert num_elts > 0
                for _ in range(num_elts):
                    symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                    symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + num_elts]
                symptom_possible_val_mapping[idx] = {a: i for i, a in enumerate(possible_values)}
                symptom_default_value_mapping[idx] = default_value
            else:
                raise ValueError(f"Type de données inconnu '{data_type}' pour le symptôme '{key}'.")

        self.obs_dtype = obs_dtype
        self.num_features = len(low_demo_values) + len(symp_low_val)
        self.symptom_to_obs_mapping = symptom_to_obs_mapping
        self.symptom_possible_val_mapping = symptom_possible_val_mapping
        self.symptom_data_types = symptom_data_types
        self.symptom_default_value_mapping = symptom_default_value_mapping
        self.categorical_integer_symptoms = categorical_integer_symptoms
        self.symptom_defaul_in_obs = [
            (symptom_to_obs_mapping[i][0], NONE_VAL)
            for i in range(num_symptoms)
        ]

    def _from_symptom_index_to_frame_index(self, symptom_index, symp_val=None):
        start, end = self.symptom_to_obs_mapping[symptom_index]
        if end - start == 1:
            return start
        if symp_val is not None and symptom_index in self.symptom_possible_val_mapping:
            val_map = self.symptom_possible_val_mapping[symptom_index]
            return start + val_map.get(str(symp_val), 0)
        return start

    def _from_symptom_index_to_inquiry_action(self, symptom_index):
        return symptom_index

    def _from_inquiry_action_to_frame_index(self, action):
        return self.symptom_to_obs_mapping[action][0], None

    def get_symptom_and_value(self, symptom_name: str):
        idx = symptom_name.find("_@_")
        if idx == -1:
            return symptom_name, None
        return symptom_name[:idx], symptom_name[idx + 3:]

    def _convert_to_aarlc_format(self, rb_data):
        age = rb_data["AGE_BEGIN"]
        sex = self.unique_genders[rb_data["GENDER"]]
        race = "white"
        ethnic = "nonhispanic"
        pathology = self.unique_pathos[rb_data["PATHOLOGY"]]
        pathology_index = self.pathology_name_2_index.get(pathology, 0)
        pathology_severity = self.pathology_severity_data[pathology_index]
        symptoms = [a for a in rb_data["SYMPTOMS"] if a in self.unique_symptoms]

        initial_symptom = None
        if self.unique_init_symptoms and "INITIAL_SYMPTOM" in rb_data and not isinstance(rb_data["INITIAL_SYMPTOM"], float):
            init_raw = rb_data["INITIAL_SYMPTOM"]
            if init_raw in self.unique_init_symptoms:
                root_name = str(init_raw).split("_@_")[0]
                initial_symptom = self.symptom_name_2_index.get(root_name)

        # Diagnostic différentiel
        differential_data = None
        if self.use_differential_diagnosis and self.max_differential_len > 0:
            diff_raw = rb_data.get("DIFFERNTIAL_DIAGNOSIS", [])
            if diff_raw:
                differential_data = {}
                for diff_entry in diff_raw:
                    if len(diff_entry) >= 2:
                        patho_name = diff_entry[0] if isinstance(diff_entry[0], str) else str(diff_entry[0])
                        proba = float(diff_entry[1])
                        if patho_name in self.pathology_name_2_index:
                            pi = self.pathology_name_2_index[patho_name]
                            differential_data[pi] = [proba, 1.0]

        out_diff = self._compute_differential_probs(differential_data)
        differential_indices, differential_probas = out_diff

        target_state = np.ones((self.state_size,), dtype=self.obs_dtype) * ABS_VAL
        if self.include_turns_in_state:
            target_state[0] = 0
        target_state = self._init_demo_features(target_state, age, sex, race, ethnic)
        binary_symptoms, present_symptoms, target_state = self.parse_target_patients(symptoms, target_state)

        return {
            "bin_sym": binary_symptoms,
            "pres_sym": present_symptoms,
            "age": age,
            "race": race,
            "sex": sex,
            "ethnic": ethnic,
            "pathology_index": pathology_index,
            "pathology_severity": pathology_severity,
            "initial_symptom": initial_symptom,
            "tgt_state": target_state,
            "differential_indices": differential_indices,
            "differential_probas": differential_probas,
        }

    def _compute_differential_probs(self, differential_data):
        if differential_data is None:
            return None, None
        n = self.num_pathos
        # ADAPTATION: Conversion prob→sommeOR→prob (double conversion DDXPlus)
        indices = np.full(n, -1, dtype=int)
        probas = np.zeros(n, dtype=np.float32)
        for pi, (proba, _) in differential_data.items():
            if 0 <= pi < n:
                indices[pi] = pi
                probas[pi] = proba
        total = probas.sum()
        if total > 0:
            probas /= total
        return indices, probas

    def _put_patients_data_in_cache(self, rb):
        patients = rb.apply(
            lambda row: self._convert_to_aarlc_format(row), axis="columns"
        ).tolist()
        self.cached_patients = {i: p for i, p in enumerate(patients)}

    def parse_target_patients(self, symptomPat, target_state):
        binary_symptoms = []
        present_symptoms = []
        symptoms_not_listed = set(self.all_symptom_names) - set(
            [self.get_symptom_and_value(s)[0] for s in symptomPat]
        )
        considered = set(symptomPat) | symptoms_not_listed

        for symptom_name in considered:
            root_name, symp_val = self.get_symptom_and_value(symptom_name)
            if root_name not in self.symptom_name_2_index:
                continue
            symptom_index = self.symptom_name_2_index[root_name]
            is_present = symptom_name in symptomPat
            data_type = self.symptom_data_types[symptom_index]
            is_antecedent = self.symptom_data[self.symptom_index_2_key[symptom_index]].get("is_antecedent", False)

            if data_type == "B" and is_present and not is_antecedent:
                binary_symptoms.append(symptom_index)

            if data_type != "B" and not is_present:
                symp_val = self.symptom_default_value_mapping.get(symptom_index)

            f_i = self._from_symptom_index_to_frame_index(symptom_index, symp_val)

            if is_present:
                if data_type == "B":
                    present_symptoms.append(symptom_index)
                else:
                    default_value = self.symptom_default_value_mapping.get(symptom_index)
                    if not (str(default_value) == str(symp_val)):
                        present_symptoms.append(symptom_index)

            if data_type == "B":
                target_state[f_i] = PRES_VAL if is_present else ABS_VAL
            elif data_type == "M":
                target_state[f_i] = PRES_VAL
            else:
                if symptom_index not in self.categorical_integer_symptoms:
                    if f_i < len(target_state):
                        target_state[f_i] = PRES_VAL
                else:
                    val_map = self.symptom_possible_val_mapping[symptom_index]
                    val_index = val_map.get(str(symp_val), 0)
                    num = len(val_map)
                    scaled = NONE_VAL + ((PRES_VAL - NONE_VAL) * (val_index + 1) / num)
                    target_state[f_i] = scaled

        return list(set(binary_symptoms)), list(set(present_symptoms)), target_state
