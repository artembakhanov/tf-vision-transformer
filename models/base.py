import tensorflow as tf
from layers import EmbeddedPatches, TransformerEncoder, MLPHead


class VisionTransformer(tf.keras.Model):
    """
    Vision transformer model.
    
    Args:
        patch_size (int): The size of one side of a square patch.
        latent_dim (int): The size of latent vectors in encoder layers.
        heads_num (int): The number of heads in MSA layers inside encoder layer.
        mlp_dim (int): The size of one hidden layer in the MLP inside encoder layer.
        encoders_num (int): The number of consequent encoder layers.
        mlp_head_dim (int): Nothing for now.
        classes_num (int): The number of classes to predict.
        dropout_rate (float): Dropout rate.
    """
    
    def __init__(self, patch_size, latent_dim, heads_num, mlp_dim, encoders_num, mlp_head_dim, classes_num, dropout_rate):
        super().__init__()
        self.emb_patches = EmbeddedPatches(patch_size, latent_dim)
        
        self.encoders = [
            TransformerEncoder(latent_dim, heads_num, mlp_dim, dropout_rate)
            for _ in range(encoders_num)
        ]
        
        self.mlp_head = MLPHead(classes_num)
        
    def call(self, input, training):
        print(training)
        x = self.emb_patches(input)
        
        for encoder in self.encoders:
            x = encoder(x)
        
        # extract class
        x = x[:, 0, :]

        x = self.mlp_head(x)
        
        return x