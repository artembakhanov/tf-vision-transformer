from models.base import VisionTransformer
from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
import json


def load_vit_model(model_dir, compiled=False):
    """Load base ViT model.
    
    Parameters
    ----------
    model_dir : string
        Saved model directory.
    compiled : bool, default=False
        Whether to load compiled version of the model.
        If one needs predicting, they should leave it False.
    """
    
    
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

def predict(model, images, labels=None):
    """Predict.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model used to predict labels.
    images : List(np.ndarray)
        Images to classify.
    labels : List(str)
        Labels to return.
    """
    
    
    if type(images) == list:
        images = tf.stack(images)
        
    predictions = model(images) 
    predictions = tf.math.argmax(predictions, axis=1)
    if labels is not None:
        predictions = [labels[pred] for pred in predictions]
    return predictions
    