"""
Model persistence utilities
Save and load trained uplift models
"""
import joblib
import json
from pathlib import Path
from datetime import datetime

def save_model(model, model_name, metadata=None):
    """
    Save trained model to models/ directory
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        metadata: Optional dict with model info
    """
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = models_dir / f"{model_name}_{timestamp}.pkl"
    
    joblib.dump(model, model_path)
    print(f"[SAVE] Model saved to {model_path}")
    
    # Save metadata
    if metadata:
        meta_path = models_dir / f"{model_name}_{timestamp}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[SAVE] Metadata saved to {meta_path}")
    
    return str(model_path)

def load_model(model_path):
    """Load trained model from file"""
    model = joblib.load(model_path)
    print(f"[LOAD] Model loaded from {model_path}")
    return model

def save_pipeline_models(models_dict):
    """
    Save all models from pipeline
    
    Args:
        models_dict: Dict with model names as keys and model objects as values
    """
    saved_paths = {}
    for name, model in models_dict.items():
        path = save_model(model, name)
        saved_paths[name] = path
    
    # Save manifest
    manifest_path = Path('models') / 'model_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(saved_paths, f, indent=2)
    print(f"[SAVE] Model manifest saved to {manifest_path}")
    
    return saved_paths
