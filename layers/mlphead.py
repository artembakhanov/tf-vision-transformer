import tensorflow as tf
from tensorflow.keras.layers import Dense

class MLPHead(tf.keras.layers.Layer):
    """
    MLP head of vision transformer.
    
    Args:
        classes_num (int): The number of classes to predict.
    """
    
    def __init__(self, classes_num):
        super().__init__()
        self.dense = Dense(classes_num)
    
    
    def call(self, input):
        return self.dense(input)