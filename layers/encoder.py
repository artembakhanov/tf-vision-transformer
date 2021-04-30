import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.activations import gelu


class TransformerEncoder(tf.keras.layers.Layer):
    """This class represents one encoder of the transformer.
    
    Parameters
    ----------
    latent_dim : int 
        The size of latent vectors in encoder layers.
    heads_num : int 
        The number of heads in MSA layers inside encoder layer.
    mlp_dim : int
        The size of one hidden layer in the MLP inside encoder layer.
    dropout_rate : float
        Dropout rate.
    """
    
    def __init__(self, 
                 latent_dim, 
                 heads_num, 
                 mlp_dim, 
                 dropout_rate):
        super().__init__()
        
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        
        self.MSA = MSA(latent_dim, heads_num, dropout_rate)
        self.MLP = MLP(latent_dim, mlp_dim, dropout_rate)
        
    def call(self, input, ret_scores=False):
        norm_input = self.ln1(input)
        
        if ret_scores:
            msa, scores = self.MSA(norm_input, ret_scores)
        else:
            msa = self.MSA(norm_input)
        
        x = msa + norm_input
        norm_msa = self.ln2(x)
        mlp = self.MLP(norm_msa)
        output = mlp + norm_msa
        
        if ret_scores:
            return output, scores
        else:
            return output


class MSA(tf.keras.layers.Layer):
    """
    Multihead self-attention.
    
    Args:
        latent_dim (int): The size of latent vectors.
        heads_num (int): The number of heads.
        dropout_rate (float): Dropout rate.
    """
    
    def __init__(self, latent_dim, heads_num, dropout_rate):
        super().__init__()
        Dh = int(latent_dim / heads_num)
        
        if int(latent_dim / heads_num) == 0:
            raise ValueError("Incorrect number of heads."
                             "Try to take smaller number.")
        
        # I decided not to write it myself since it is several matrix 
        # multiplications and I might make a mistake somewhere 
        # or do it inefficiently.
        self.mha = MultiHeadAttention(
            heads_num, 
            Dh, 
            attention_axes=(1,),
            dropout=dropout_rate,
        )
    
    
    def call(self, input, ret_scores=False):
        return self.mha(input, input, return_attention_scores=ret_scores)
    

class MLP(tf.keras.layers.Layer):
    """
    A simple two-layer perceptron used in encoders.
    
    Args:
        latent_dim (int): The size of latent vectors.
        mlp_dim (int): The size of a hidden layer.
        dropout_rate (float): Dropout rate.
    """
    
    def __init__(self, latent_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.dense1 = Dense(mlp_dim)
        self.dropout1 = Dropout(dropout_rate)
        
        self.dense2 = Dense(latent_dim)
        self.dropout2 = Dropout(dropout_rate)
    
    
    def call(self, input):
        x = self.dense1(input)
        x = gelu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        output = self.dropout2(x)
        
        return output