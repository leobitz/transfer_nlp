#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
random.seed(8080)
import numpy as np
np.random.seed(8080)
import tensorflow as tf
tf.random.set_seed(8080)

from data_gen import *
import time
import preprocess as pre
import argparse


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int,  default=128)
parser.add_argument("--char_embed_size", type=int,  default=32)
parser.add_argument("--feat_embed_size", type=int, default=32)
parser.add_argument("--hidden_size", type=int, default=265)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--file_name", type=str, default='wol-14')
args = parser.parse_args()


# In[ ]:


char2int, feat2val, max_r, max_w = pre.process([args.file_name])
data = pre.convert(char2int, feat2val, max_r, max_w, langs=[args.file_name], for_cnn=True)
clean_data = pre.convert(char2int, feat2val, max_r, max_w, langs=['wol'], train_set=False, for_cnn=True)
gen_data = pre.convert(char2int, feat2val, max_r, max_w, langs=[args.file_name], train_set=False, for_cnn=True)
int2char = {val: key for val, key in enumerate(char2int)}


# In[ ]:


batch_size = args.batch_size
max_root = max_r + 2
max_word = max_w + 2
n_feature = data[1].shape[1]
hidden_size = args.hidden_size
feat_embed_size = args.feat_embed_size
char_embed_size = args.char_embed_size
EPOCHS = args.epochs
n_batches = len(data[0]) // batch_size
print("Total Data: {0} Total Batches {1}".format(len(data[0]), n_batches))


# In[ ]:


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, feat_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform', name="encoder_gru")
        self.fc1 = tf.keras.layers.Dense(feat_units, activation="relu", name="feature_output")
        self.fc2 = tf.keras.layers.Dense(enc_units, activation="relu", name="state_out")
        
    def call(self, w, f, hidden):
        output, state = self.gru(w, initial_state=hidden)
        feat = self.fc1(f)
        state = tf.concat([state, feat], axis=1)
        state = self.fc2(state)
        return output, state, feat

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units), dtype=tf.float32)


# In[ ]:


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[ ]:


class Decoder(tf.keras.Model):
    def __init__(self, dec_units, output_size, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform', name="decoder_gru")
        self.fc = tf.keras.layers.Dense(output_size, activation="softmax")

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, feat):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = tf.concat([context_vector, x, feat], axis=-1)
        x = tf.expand_dims(x, 1)
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)
        return x, state#, attention_weights


# In[ ]:


def predict(encoder, decoder, inputs, n_steps):
    # encode
    root, feat = inputs[0], inputs[1]
    enc_hidden = encoder.initialize_hidden_state()
    enc_output, state, feat = encoder(inputs[0], inputs[1], enc_hidden)
    
    start_word = '<'
    start_mat = pre.word_to_matrix(start_word, char2int, 1, ' ')
    
    target_seq = np.zeros((root.shape[0], len(char2int)), dtype=np.float32) + np.array(start_mat, dtype=np.float32)
    outputs = list()
    for t in range(n_steps):
        # predict next char
        target_seq, state = decoder(target_seq, state, enc_output, feat)
        
        outputs.append(target_seq)
    return np.stack(outputs)


# In[ ]:


decoder = Decoder(hidden_size, len(char2int), batch_size)
encoder = Encoder(hidden_size, feat_embed_size, batch_size)


# In[ ]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()

def loss_function(real, pred):
    loss_ = loss_object(real, pred)

    return tf.reduce_mean(loss_)


# In[ ]:


@tf.function
def train_step(root, feature, dec_input, target, enc_hidden):
    loss = 0
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden, feat = encoder(root, feature, enc_hidden)

        dec_hidden = enc_hidden

        for t in range(target.shape[1]):
            predictions, dec_hidden = decoder(dec_input[:, t], dec_hidden, enc_output, feat)
            loss += loss_function(target[:, t], predictions)

        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


# In[ ]:


def test_model(test_data):
    test_n_batches, test_batch_size =  int(test_data[0].shape[0] / batch_size), batch_size  
    # print(test_n_batches * test_batch_size)
    test_gen = pre.gen(test_data, batch_size, shuffle=False)
    # shows sample examples and calculates accuracy
    test_batches = len(test_data[0]) // batch_size
    total, correct = 0, 0
    in_word = 0
    sims = []
    for b in range(test_batches - 1):
        # get data from test data generator
        [root, feat, dec_in], y = next(test_gen)
        pred = predict(encoder, decoder, [root, feat], max_word)
        for k in range(pred.shape[1]):
            indexes = pred[:, k]#.argmax(axis=1)
            r = ''.join(pre.matrix_to_word(root[k], int2char)).strip()[1:-1]
            w = ''.join(pre.matrix_to_word(dec_in[k], int2char)).strip()[1:-1]
            t = ''.join(pre.matrix_to_word(indexes, int2char)).strip()[:-1]
            if w == t:
                correct += 1
    #         else:
    #             print(r, w, t)


        total += batch_size
        return float(correct)/float(total)*100.0


# In[ ]:


gen = pre.gen(data, batch_size)


# In[ ]:


for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for step in range(n_batches):
        [root, feat, dec_in], y = next(gen)
        batch_loss = train_step(root, feat, dec_in, y, enc_hidden)
        total_loss += batch_loss

#         if step % (n_batches // 1) == 0:
#             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
#                                                      step,
#                                                      batch_loss.numpy()))
    elaps = time.time() - start
    clean_accuracy = test_model(clean_data)
    gen_accuracy = test_model(gen_data)
    print('Epoch {} Loss {:.4f} Gen Accuracy {:.4f} Clean Accuracy {:.4f} Time {:.4f}'.format(epoch + 1,total_loss / n_batches, gen_accuracy, clean_accuracy, elaps))
    


# In[ ]:




