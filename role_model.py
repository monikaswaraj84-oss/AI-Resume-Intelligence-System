"""Loads the job-role classifier without serve-time retraining."""
import logging
import os
import joblib

from config import MODEL_PATH

logger = logging.getLogger(__name__)

_state = {"model": None, "status": "not loaded"}


def load_model():
    if _state["model"] is not None:
        return _state["model"]
    if not os.path.exists(MODEL_PATH):
        _state["status"] = "model file not found"
        logger.warning("Model file not found at %s. Please run: python ml_model/train_model.py", MODEL_PATH)
        return None
    try:
        _state["model"] = joblib.load(MODEL_PATH)
        _state["status"] = "loaded"
    except Exception as exc:
        _state["status"] = f"load failed: {exc}"
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
    return _state["model"]


def status():
    load_model()
    return _state["status"]


def predict_role(resume_text, min_confidence=0.40):
    """Return (role, confidence 0-100). Low confidence -> 'Uncertain (closest: ...)'."""
    model = load_model()
    if model is None:
        return "Not available", 0.0
    from ml_model.train_model import prepare_text
    text = prepare_text(resume_text)
    if not text:
        return "Not available", 0.0
    try:
        proba = model.predict_proba([text])[0]
        idx = int(proba.argmax())
        role, conf = str(model.classes_[idx]), float(proba[idx])
    except Exception as exc:
        logger.warning("predict_role failed: %s", exc)
        return "Prediction failed", 0.0
    if conf < min_confidence:
        return f"Uncertain (closest: {role})", round(conf * 100, 1)
    return role, round(conf * 100, 1)
