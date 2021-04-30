from const import DATASETS
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


def split_dataset(x, batch_size):
    # taken from: https://stackoverflow.com/a/64100245/8321904
    return np.split(x, np.arange(batch_size, len(x), batch_size))[-1].shape
    
    
def load_data(dataset, batch_size=128):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}. Please use one of {DATASETS}.")
        
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
        
        train_ds = Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        test_ds = Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        
        return train_ds, test_ds
    
    
    elif dataset == "fruits-360":
        dataset_dir = f"./datasets/{dataset}"
        if not os.path.isdir(dataset_dir):
            raise RuntimeError(f"The {dataset} cannot be found. "
                               f"Please download it and unpack into datasets directory. "
                               f"Do not forget to mount the directory to this container. ")
            
            
        train_dir = os.path.join(dataset_dir, "Training")
        test_dir = os.path.join(dataset_dir, "Test")
        
        train_ds = ImageDataGenerator(
            rescale= 1.0 / 255,
            horizontal_flip=True)

        test_ds = ImageDataGenerator(rescale=1.0 / 255)
        
        train_ds = train_ds.flow_from_directory(train_dir, 
                                     target_size=(100, 100),
                                     batch_size=batch_size,
                                     class_mode="sparse")
        
        test_ds = test_ds.flow_from_directory(test_dir, 
                                     target_size=(100, 100),
                                     batch_size=batch_size,
                                     class_mode="sparse")
        
        return train_ds, test_ds
            
def load_labels(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}. Please use one of {DATASETS}.")
        
    if dataset == "mnist":
        return list("0123456789")
    
    elif dataset == "fruits-360":
        dataset_dir = f"./datasets/{dataset}"
        if not os.path.isdir(dataset_dir):
            raise RuntimeError(f"The {dataset} cannot be found. "
                               f"Please download it and unpack into datasets directory. "
                               f"Do not forget to mount the directory to this container. ")
        
        labels = sorted(os.listdir("datasets/fruits-360/Test"))
        
        assert len(labels) == 131
        
        return labels