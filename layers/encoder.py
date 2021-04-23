import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.activations import gelu


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, heads_num, mlp_dim, dropout_rate):
        super().__init__()
        
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        
        self.MSA = MSA(latent_dim, heads_num)
        self.MLP = MLP(latent_dim, mlp_dim, dropout_rate)
        
    def call(self, input):
        norm_input = self.ln1(input)
        msa = self.MSA(norm_input)
        x = msa + norm_input
        norm_msa = self.ln2(x)
        mlp = self.MLP(norm_msa)
        output = mlp + norm_msa
        
        return output


class MSA(tf.keras.layers.Layer):
    def __init__(self, D, heads):
        super().__init__()
        Dh = int(D / heads)
        
        if int(D / heads) == 0:
            raise ValueError("Incorrect number of heads."
                             "Try to take smaller number.")
        
        # I decided not to write it myself since it is several matrix 
        # multiplications and I might make a mistake somewhere 
        # or do it inefficiently.
        self.mha = MultiHeadAttention(heads, Dh, attention_axes=(2,))
    
    
    def call(self, input):
        return self.mha(input, input)
    

class MLP(tf.keras.layers.Layer):
    def __init__(self, latent_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.dense1 = Dense(mlp_dim)
        self.dropout1 = Dropout(dropout_rate)
        
        self.dense2 = Dense(latent_dim)
        self.dropout2 = Dropout(dropout_rate)
    
    
    def call(self, input, training):
        x = self.dense1(input)
        x = gelu(x)
        x = self.dropout1(x, training)
        x = self.dense2(x)
        x = gelu(x)
        output = self.dropout2(x, training)
        
        return output