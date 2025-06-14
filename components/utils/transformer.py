
# tensorflow.py
# In this file, we build the transformer architecture from scratch using tensorflow

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout, LayerNormalization
import numpy as np

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Multi-Head Attention
# splits the input tensor into three 
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = Dense(d_model) # query weights Wq
        self.wk = Dense(d_model) # key weights Wk
        self.wv = Dense(d_model) # value weights Wv
        self.dense = Dense(d_model) # Linear feed forward network output weights Wo

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q) # QWq
        k = self.wk(k) # KWk
        v = self.wv(v) # VWv
        q = self.split_heads(q, batch_size) # split QWq into h heads
        k = self.split_heads(k, batch_size) # split KWk into h heads
        v = self.split_heads(v, batch_size) # split VWv into h heads
        
        attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask) # compute attention on each head
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model)) # concatenate all attentions
        output = self.dense(attention) # concat*Wo
        return output
    
    # attention function
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
    
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    

# Positionwise Feed Forward Network
# The positionwise feed forward network is composed of two feed forward networks
# the first one has a relu activation function
class PositionwiseFeedforward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedforward, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    

# Transformer Block
# A transformer block is composed of a multihead attention layer, a layer normalization layer,
# a point wise feed forward network and a layer normalization layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedforward(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Encoder stack
# The encoder stack is composed of :
# Embedding : Embed tokens in high dimensional space
# Positional Encoding : Encode tokens position in high dimensional space
# stack of transformer blocks (n)   
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout = Dropout(dropout_rate)
        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x
    

# Decoder stack
# The decoder stack is composed of :
# Embedding : embed outputs (shifted right) in high dimensional space
# Positional encoding : encode tokens position in high dimensional space
# stack of transformer blocks (n)  
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout = Dropout(dropout_rate)
        self.dec_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training=training, mask=look_ahead_mask)
        return x, attention_weights
    

# Transformer
# This is a modified transformer architecture
# Our transformer is composed just an encoder layer and a dense linear layer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               vocab_size, maximum_position_encoding, dropout_rate)
        # self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
        #                        target_vocab_size, maximum_position_encoding, dropout_rate)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.classifier = tf.keras.Sequential([
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
        inp = inputs
        enc_output = self.encoder(inp, training=training, mask=padding_mask)
        pooled = self.global_avg_pool(enc_output)
        # dec_output, _ = self.decoder(tar, enc_output, training=training, 
        #                             look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        # final_output = self.final_layer(dec_output)
        predictions = self.classifier(pooled, training=training)

        return predictions
    
    def create_padding_mask(self, seq):
        """Create padding mask for input sequences"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]












