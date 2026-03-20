import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
filepath = os.path.join(BASE_DIR, "data", "processed", "release_evidences.json")

with open(filepath, "r", encoding="utf-8") as f:
    evidences = json.load(f)

for key, data in evidences.items():
    name = data.get("name", key).replace("_", " ")
    is_atcd = data.get("is_antecedent", False)
    data_type = data.get("data_type", "B")

    # Formulation de la question
    if data_type == "B":
        if is_atcd:
            q_fr = f"Avez-vous cet antécédent : {name} ?"
            q_en = f"Do you have this medical history: {name}?"
        else:
            q_fr = f"Ressentez-vous ce symptôme : {name} ?"
            q_en = f"Do you experience this symptom: {name}?"
        
        prop_fr = [{"label": "Oui", "value": 1}, {"label": "Non", "value": -1}, {"label": "Ne sait pas", "value": 0}]
        prop_en = [{"label": "Yes", "value": 1}, {"label": "No", "value": -1}, {"label": "Don't know", "value": 0}]
    
    elif data_type in ["C", "M"]:
        q_fr = f"Concernant '{name}', quelle est votre situation ?"
        q_en = f"Regarding '{name}', what is your situation?"
        
        possible_vals = data.get("possible-values", [])
        prop_fr = [{"label": str(v), "value": v} for v in possible_vals] + [{"label": "Ne sait pas", "value": 0}]
        prop_en = [{"label": str(v), "value": v} for v in possible_vals] + [{"label": "Don't know", "value": 0}]

    data["question_fr"] = q_fr
    data["question_en"] = q_en
    data["propositions_fr"] = prop_fr
    data["propositions_en"] = prop_en

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(evidences, f, indent=4, ensure_ascii=False)

print(f"✅ Fichier '{filepath}' mis à jour avec les questions et propositions.")
