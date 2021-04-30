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
        self.config = {
            "patch_size": patch_size,
            "latent_dim": latent_dim,
            "heads_num": heads_num,
            "mlp_dim": mlp_dim,
            "encoders_num": encoders_num,
            "mlp_head_dim": mlp_head_dim,
            "classes_num": classes_num,
            "dropout_rate": dropout_rate
        }
        self.emb_patches = EmbeddedPatches(patch_size, latent_dim, dropout_rate)
        
        self.encoders = [
            TransformerEncoder(latent_dim, heads_num, mlp_dim, dropout_rate)
            for _ in range(encoders_num)
        ]
        
        self.mlp_head = MLPHead(classes_num)
        
    def call(self, input, ret_scores=False):
        x = self.emb_patches(input)
        
        all_scores = []
        
        for encoder in self.encoders:
            x = encoder(x, ret_scores)
            if ret_scores:
                x, scores = x
                all_scores.append(scores)
                
        
        # extract class
        x = x[:, 0, :]

        x = self.mlp_head(x)
        
        if ret_scores:
            return x, all_scores
        else:
            return x
        