from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, TimeDistributed
from keras.layers import Concatenate, Flatten
from keras.layers import GRU, Conv2D, MaxPooling2D
from keras.layers import Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
import keras
import numpy as np


def conv_model(n_input, n_output, n_feature, n_units, feat_units = 5):
    root_word_input = Input(shape=(15, 28, 1), name="root_word_input")
    feature_input = Input(shape=(n_feature,), name="word_feature_input")
    feat_out = Dense(feat_units, activation="relu", name="feature_output")(feature_input)
    
    x = Conv2D(20, (5, 5), padding='same', activation='relu', name="cnn")(root_word_input)
    x = MaxPooling2D(3, 3, name="pooling")(x)

    flat_output = Flatten(name="flatten")(x)
#     x = Concatenate()([x, feat_out])
#     state_h = Dense(n_dec_units, activation='relu')(x)
    x = Dense(n_units - feat_units, activation='relu', name="cnn_encoder")(flat_output)
    
    state_h = Concatenate(name="concatnate")([x, feat_out])
    
    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    decoder_gru = GRU(n_units, return_sequences=True, return_state=True, name="decoder_gru")
    decoder_outputs, _= decoder_gru(decoder_inputs, initial_state=state_h)
    
    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([root_word_input, decoder_inputs, feature_input], decoder_outputs)
    encoder_model = Model([root_word_input, feature_input], state_h)
    
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_outputs, state_h= decoder_gru(decoder_inputs, initial_state=decoder_state_input_h)

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input_h], [decoder_outputs, state_h])

    return model, encoder_model, decoder_model

def predict(infenc, infdec, source, feat, n_steps, cardinality):
    # encode
    state = infenc.predict([source, feat])
    # start of sequence input
    start = [0.0 for _ in range(cardinality)]
#     start[0] = 1
    target_seq = np.array(start).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h= infdec.predict([target_seq, state])
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = h
        # update target sequence
        target_seq = yhat
    return np.array(output)

