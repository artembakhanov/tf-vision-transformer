import tensorflow as tf
import warnings
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import RandomNormal


class EmbeddedPatches(tf.keras.layers.Layer):
    """
    This layers extracts patches of size P and 
    projects them to a latent space of size D;
    Add class and position embeddings.
    
    Args:
        patch_size (int): The size of one side of a square patch.
        latent_dim (int): The size of latent vectors in encoder layers.
            All the patches will be projected to this dimension.
        dropout_rate (float): Dropout rate.
    """
    
    def __init__(self, patch_size, latent_dim, dropout_rate):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.projection = Dense(self.latent_dim, use_bias=False)
        self.class_emb = self.add_weight(
            name="class_emb",
            shape=(1, 1, self.latent_dim),
            initializer="zeros"
        )
        self.broadcast_class_emb = None
        
        self.position_emb = None
        
        self.dropout = Dropout(dropout_rate)
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Input tensor should have 4 dimensions.")
           
        
        (_, H, W, C) = input_shape
        self.N = int(H * W / (self.patch_size ** 2))
        
        if H * W % self.patch_size ** 2 != 0:
            warnings.warn(
                f"It is recommended to have valid padding.\n"
                f"Only {self.N * self.patch_size}x{self.N * self.patch_size} pixels will be considered.\n" 
                f"Try to put another value of P if you are afraid of this warning."
            )
           
        self.C = input_shape[3]
        
        
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.N + 1, self.latent_dim),
            initializer=RandomNormal(stddev=0.02) # FROM BERT
        )
    
        
    def call(self, input):
        batch_size = tf.shape(input)[0]
        patches = tf.image.extract_patches(input, 
                                        sizes=[1, self.patch_size, self.patch_size, 1], 
                                        strides=[1, self.patch_size, self.patch_size, 1], 
                                        rates=[1, 1, 1, 1], 
                                        padding="VALID")
        
        patches = self.projection(patches)
        
        # remove one dimension
        patches = tf.reshape(patches, (batch_size, -1, self.latent_dim))
        
        # add embedding
        bc_class_emb = tf.broadcast_to(self.class_emb, 
                                            [batch_size, 1, self.latent_dim])
        patches = tf.concat((bc_class_emb, patches), axis=1)
        
        # add position embeddings
        bc_pos_emb = tf.broadcast_to(self.pos_emb,
                                          [batch_size, self.N + 1, self.latent_dim])
        patches = patches + bc_pos_emb
        
        # dropout
        patches = self.dropout(patches)
        
        return patches

    