import tensorflow as tf
import warnings
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal


class EmbeddedPatches(tf.keras.layers.Layer):
    """
    This layers extracts patches of size P and 
    projects them to a latent space of size D;
    Add class and position embeddings.
    """
    
    def __init__(self, P, D):
        super().__init__()
        self.P = P
        self.D = D
        self.projection = Dense(self.D, use_bias=False)
        self.class_emb = self.add_weight(
            name="class_emb",
            shape=(1, 1, self.D),
            initializer="zeros"
        )
        self.broadcast_class_emb = None
        
        self.position_emb = None
        
    def build(self, input_shape):
        print(input_shape)
        if len(input_shape) != 4:
            raise ValueError("Input tensor should have 4 dimensions.")
           
        
        (_, H, W, C) = input_shape
        self.N = int(H * W / (self.P ** 2))
        
        if H * W % self.P ** 2 != 0:
            warnings.warn(
                f"It is recommended to have valid padding.\n"
                f"Only {self.N * self.P}x{self.N * self.P} pixels will be considered.\n" 
                f"Try to put another value of P if you are afraid of this warning."
            )
           
        self.C = input_shape[3]
        
        
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.N + 1, self.D),
            initializer=RandomNormal(stddev=0.02) # FROM BERT
        )
    
        
    def call(self, input):
        batch_size = tf.shape(input)[0]
        patches = tf.image.extract_patches(input, 
                                        sizes=[1, self.P, self.P, 1], 
                                        strides=[1, self.P, self.P, 1], 
                                        rates=[1, 1, 1, 1], 
                                        padding="VALID")
        
        patches = self.projection(patches)
        
        # remove one dimension
        patches = tf.reshape(patches, (batch_size, -1, self.D))
        
        # add embedding
        bc_class_emb = tf.broadcast_to(self.class_emb, 
                                            [batch_size, 1, self.D])
        patches = tf.concat((bc_class_emb, patches), axis=1)
        
        # add position embeddings
        bc_pos_emb = tf.broadcast_to(self.pos_emb,
                                          [batch_size, self.N + 1, self.D])
        patches = patches + bc_pos_emb
        
        return patches

    