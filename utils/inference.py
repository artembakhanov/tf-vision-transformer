from models.base import VisionTransformer
from tensorflow.keras.models import load_model
from pathlib import Path
import json


def load_vit_model(model_dir, compiled=False):
    model_dir = Path(model_dir)
    config_file = model_dir / "config.json"
    
    if not model_dir.is_dir() or not config_file.is_file():
        raise ValueError(f"{model_dir} is not a valid model directory.")
    
    if compiled:
        model = load_model(model_dir / "final.model")
    else:
        with open(config_file) as f:
            model = VisionTransformer(**(json.loads(f.read())))
        model.load_weights(model_dir / "final.weights")
    
    
    return model