import os
import joblib

def ensure_dir(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, filepath):
    """Save a machine learning model or pipeline using joblib."""
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load a saved model or pipeline."""
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"File not found: {filepath}")
