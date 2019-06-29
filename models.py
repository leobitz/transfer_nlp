from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.layers import GRU, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.utils.vis_utils import plot_model
import tensorflow.keras as keras
import numpy as np


def conv_model(root_h, root_w, decoder_input, decoder_output, n_feature, hidden_size, feat_units = 15):
    root_word_input = Input(shape=(root_h, root_w, 1), name="root_word_input")
    feature_input = Input(shape=(n_feature,), name="word_feature_input")
    feat_out = Dense(feat_units, activation="relu", name="feature_output")(feature_input)
    
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(root_word_input)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    # x = Concatenate()([x, feat_out])
    # state_h = Dense(hidden_size, activation='relu')(x)
    x = Dense(hidden_size - feat_units, activation='relu')(x)
    
    state_h = Concatenate()([x, feat_out])
    
    decoder_inputs = Input(shape=(None, decoder_input), name="target_word_input")
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name="decoder_gru")
    decoder_outputs, _= decoder_gru(decoder_inputs, initial_state=state_h)
    
    decoder_dense = Dense(decoder_input, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([root_word_input, feature_input, decoder_inputs], decoder_outputs)
    encoder_model = Model([root_word_input, feature_input], state_h)
    
    decoder_state_input_h = Input(shape=(hidden_size,))
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

