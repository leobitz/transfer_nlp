#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from data_gen import *
import time
import preprocess as pre
import argparse


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int,  default=128)
parser.add_argument("--char_embed_size", type=int,  default=32)
parser.add_argument("--feat_embed_size", type=int, default=16)
parser.add_argument("--hidden_size", type=int, default=265)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--file_name", type=str, default='wol-14')
args = parser.parse_args()


# In[ ]:


char2int, feat2val, max_r, max_w = pre.process([args.file_name])
# print(feat2val)
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
        self.cnn = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.pool = tf.keras.layers.MaxPool2D(2, 2)
        self.flat = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(feat_units, activation="relu", name="feature_output")
        self.fc2 = tf.keras.layers.Dense(enc_units, activation="relu", name="state_out")
        
    def call(self, w, f):
        x = self.cnn(w)
        x = self.pool(x)
        state = self.flat(x)
        feat = self.fc1(f)
        #state = tf.concat([x, feat], axis=1)
        state = self.fc2(state)
        return state, feat


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


    def call(self, x, feat, hidden):
        # enc_output shape == (batch_size, max_length, hidden_size)
        x = tf.concat([x, feat], axis=-1)
        x = tf.expand_dims(x, 1)
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)
        return x, state#, attention_weights


# In[ ]:


def predict(encoder, decoder, inputs, n_steps):
    # encode
    root, feat = inputs[0], inputs[1]
    state, feat = encoder(inputs[0], inputs[1])
    
    start_word = '<'
    start_mat = pre.word_to_matrix(start_word, char2int, 1, ' ')
    
    target_seq = np.zeros((root.shape[0], len(char2int)), dtype=np.float32) + np.array(start_mat, dtype=np.float32)
    outputs = list()
    for t in range(n_steps):
        # predict next char
        target_seq, state = decoder(target_seq, feat, state)
        
        outputs.append(target_seq)
    return np.stack(outputs)


# In[ ]:


decoder = Decoder(hidden_size, len(char2int), batch_size)
encoder = Encoder(hidden_size, feat_embed_size, batch_size)

# x = np.random.randn(10, 15, 28,1)
# f = np.random.randn(10, 32)
# h = tf.cast(np.random.randn(10, 256), tf.float64)
# x, f = encoder(x, f)
# decoder(x, f, None)

# In[ ]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()

def loss_function(real, pred):
    loss_ = loss_object(real, pred)

    return tf.reduce_mean(loss_)


# In[ ]:


@tf.function
def train_step(root, feature, dec_input, target):
    loss = 0
    
    with tf.GradientTape() as tape:
        enc_hidden, feat = encoder(root, feature)
        # print(enc_hidden.shape)
        dec_hidden = enc_hidden

        for t in range(target.shape[1]):
            predictions, dec_hidden = decoder(dec_input[:, t], feat, dec_hidden)
            loss += loss_function(target[:, t], predictions)

        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


# In[ ]:


def test_model(test_data, log=False):
    test_n_batches, test_batch_size =  int(test_data[0].shape[0] / batch_size), batch_size  
    #print(test_n_batches * test_batch_size)
    test_gen = pre.gen(test_data, batch_size, shuffle=False)
    # shows sample examples and calculates accuracy
    test_batches = len(test_data[0]) // batch_size
    total, correct = 0, 0
    in_word = 0
    sims = []
    for b in range(test_batches - 1):
        # get data from test data generator
        [root, feat, dec_in], y = next(test_gen)
        root = np.expand_dims(root, axis=3)
        pred = predict(encoder, decoder, [root, feat], max_word)
        for k in range(pred.shape[1]):
            indexes = pred[:, k]#.argmax(axis=1)
            r = ''.join(pre.matrix_to_word(root[k], int2char)).strip()[1:-1]
            w = ''.join(pre.matrix_to_word(dec_in[k], int2char)).strip()[1:-1]
            t = ''.join(pre.matrix_to_word(indexes, int2char)).strip()[:-1]
            if w == t:
                correct += 1
            else:
                if log:
                    print(r, w, t)


        total += batch_size
        return float(correct)/float(total)*100.0


# In[ ]:


gen = pre.gen(data, batch_size)


# In[ ]:


for epoch in range(EPOCHS):
    start = time.time()

    # enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for step in range(n_batches):
        [root, feat, dec_in], y = next(gen)
        root = np.expand_dims(root, axis=3)
        batch_loss = train_step(root, feat, dec_in, y)
        total_loss += batch_loss

#         if step % (n_batches // 1) == 0:
#             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
#                                                      step,
#                                                      batch_loss.numpy()))

    clean_accuracy = test_model(clean_data)
    gen_accuracy = test_model(gen_data)
    elaps = time.time() - start
    print('Epoch {} Loss {:.4f} Gen Accuracy {:.4f} Clean Accuracy {:.4f} Time {:.4f}'.format(epoch + 1,total_loss / n_batches, gen_accuracy, clean_accuracy, elaps))
    

# In[ ]:




