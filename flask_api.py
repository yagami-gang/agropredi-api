"""

API Flask (squelette) pour AgroPRedi.



Règles:

- Pas d'entraînement ici.

- On charge uniquement un modèle déjà entraîné: model_pepper.pth

- On charge classes.json pour conserver l'ordre des classes.

- On applique exactement les transforms de validation/test.

"""



from __future__ import annotations



import io

import json

import os

from pathlib import Path



import torch

import torch.nn as nn

from flask import Flask, jsonify, request

from PIL import Image

from torchvision import transforms

from torchvision.models import ResNet18_Weights, resnet18



app = Flask(__name__)

app.json.ensure_ascii = False

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB



HERE = Path(__file__).resolve().parent

MODEL_PATH = HERE / "model_pepper.pth"

CLASSES_PATH = HERE / "classes.json"



_model: nn.Module | None = None

_classes: list[str] | None = None

_device: torch.device | None = None





def _build_eval_transforms(img_size: int = 224) -> transforms.Compose:

    """Transforms de validation/test (doit matcher train_model.py)."""

    imagenet_mean = [0.485, 0.456, 0.406]

    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose(

        [

            transforms.Resize((img_size, img_size)),

            transforms.ToTensor(),

            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),

        ]

    )





def _load_classes() -> list[str]:

    """Charger la liste des classes dans l'ordre d'entraînement."""

    if not CLASSES_PATH.exists():

        raise FileNotFoundError(f"classes.json introuvable: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:

        payload = json.load(f)

    classes = payload.get("classes")

    if not isinstance(classes, list) or not classes:

        raise ValueError("classes.json invalide: champ 'classes' manquant ou vide")

    return [str(c) for c in classes]





def _load_model() -> tuple[nn.Module, list[str], torch.device]:

    """Charger le modèle entraîné (.pth) + classes.json."""

    global _model, _classes, _device



    if _model is not None and _classes is not None and _device is not None:

        return _model, _classes, _device



    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Modèle introuvable: {MODEL_PATH}. Lancez d'abord ai_api/train_pepper_model.py"

        )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = _load_classes()



    # Important: l'API doit reconstruire exactement l'architecture utilisée.

    weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, len(classes))



    checkpoint = torch.load(MODEL_PATH, map_location=device)

    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None

    if state_dict is None:

        raise ValueError("Checkpoint invalide: clé 'state_dict' manquante")

    model.load_state_dict(state_dict)



    model.to(device)

    model.eval()



    _model, _classes, _device = model, classes, device

    return model, classes, device





def _parse_plant_disease(class_name: str) -> tuple[str | None, str | None]:

    """Extraire (plant, disease) depuis une classe PlantVillage typique."""

    # Ex: Pepper__bell___healthy, Pepper__bell___Bacterial_spot

    if not class_name:

        return None, None

    parts = class_name.split("_", 1)

    plant = parts[0] if parts else None

    disease = parts[1].replace("_", " ") if len(parts) > 1 else None

    return plant, disease





def _to_laravel_schema(class_name: str, confidence_01: float, prob_dict: dict, *, status: str = "ok", message: str | None = None):

    """Adapter la prédiction au schéma métier attendu par Laravel."""

    plant, disease = _parse_plant_disease(class_name)

    # Plante dynamique selon la classe détectée
    plant_name = "Pepper" if plant and "pepper" in plant.lower() else (plant if plant else "Unknown")

    # Etat
    health_status = "Healthy" if "healthy" in class_name.lower() else "Diseased"

    # Maladie (si sain, on renvoie 'Healthy' / 'Sain' côté métier)
    if health_status == "Healthy":
        disease_name = "Healthy"
    else:
        # Normaliser le nom de maladie pour correspondre à Laravel
        # "bell Bacterial spot" → "Bacterial Spot"
        disease = (disease or "Unknown").strip()
        if "bacterial" in disease.lower() and "spot" in disease.lower():
            disease_name = "Bacterial Spot"
        else:
            disease_name = disease

    # Champs attendus par la BD Laravel (non null)
    risk_level = "Low" if health_status == "Healthy" else "High"
    conseils = (
        [
            "Surveiller régulièrement la parcelle",
            "Maintenir une bonne hygiène culturale",
        ]
        if health_status == "Healthy"
        else [
            "Reprendre une photo nette de la feuille",
            "Isoler si possible les plants atteints",
            "Consulter un technicien/agronome si les symptômes persistent",
        ]
    )

    # Confiance : EXACTEMENT max(softmax) * 100
    confidence_percent = round(float(confidence_01) * 100.0, 2)

    payload = {
        "plant": plant_name,
        "disease": disease_name,
        "status": health_status,
        "confidence": confidence_percent,
        "softmax": prob_dict,
        "class": class_name,
        "risk_level": risk_level,
        "recommendations": conseils,
    }

    if message:
        payload["message"] = message

    return payload





@app.route("/health", methods=["GET"])

def health():

    """Statut minimal + présence des artefacts."""

    return jsonify(

        {

            "status": "ok",

            "model_exists": MODEL_PATH.exists(),

            "classes_exists": CLASSES_PATH.exists(),

        }

    ), 200





@app.route("/predict", methods=["POST"])

def predict():

    """Prédiction sur une image envoyée en multipart/form-data (champ: file)."""

    try:

        if "file" not in request.files:

            return jsonify(

                _to_laravel_schema(

                    class_name="",

                    confidence_01=0.0,

                    prob_dict={},

                    status="error",

                    message="Champ 'file' manquant",

                )

            ), 400



        f = request.files["file"]

        if not f or f.filename == "":

            return jsonify(

                _to_laravel_schema(

                    class_name="",

                    confidence_01=0.0,

                    prob_dict={},

                    status="error",

                    message="Nom de fichier vide",

                )

            ), 400



        model, classes, device = _load_model()

        tf = _build_eval_transforms(img_size=224)



        img_bytes = f.read()

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        x = tf(img).unsqueeze(0).to(device)



        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(logits, dim=1)

            conf, idx = torch.max(probs, dim=1)



        idx_i = int(idx.item())

        confidence = float(conf.item())

        class_name = classes[idx_i]

        # Vecteur de probabilités pour toutes les classes
        prob_vector = probs.cpu().numpy()[0].tolist()

        # Créer le dictionnaire softmax pour la réponse
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, prob_vector)}

        response = _to_laravel_schema(class_name=class_name, confidence_01=confidence, prob_dict=prob_dict, status="ok")



        # Bonus: message utilisateur si confiance < 50%

        if response.get("confidence", 0) < 50.0:

            response["message"] = "Confiance faible: résultat à vérifier (reprendre une photo nette)."



        return jsonify(response), 200



    except Exception as e:

        return jsonify(

            _to_laravel_schema(

                class_name="",

                confidence_01=0.0,

                prob_dict={},

                status="error",

                message=str(e),

            )

        ), 500





@app.route("/info", methods=["GET"])

def info():

    """Infos sur le modèle chargé (ou chargeable)."""

    try:

        classes = _load_classes() if CLASSES_PATH.exists() else []

        return jsonify(

            {

                "architecture": "resnet18",

                "model_path": str(MODEL_PATH),

                "classes_path": str(CLASSES_PATH),

                "num_classes": len(classes),

                "classes": classes,

                "input_size": [224, 224],

                "target_plant": "Piment",

            }

        ), 200

    except Exception as e:

        return jsonify({"error": "info_error", "message": str(e)}), 500





@app.errorhandler(413)

def file_too_large(_e):

    return jsonify({"error": "file_too_large", "message": "Taille maximale: 16MB"}), 413





@app.errorhandler(404)

def not_found(_e):

    return jsonify({"error": "not_found", "message": "Endpoints: /health, /info, /predict"}), 404





if __name__ == "__main__":

    # Pas d'exécution automatique demandée par la spec; ce main est juste pour usage manuel.

    # Exemple: python ai_api/flask_api.py

    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
