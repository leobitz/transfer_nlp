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
    decoder_model = Model([decoder_inputs, decoder_state_input_h], [
                          decoder_outputs, state_h])

    return model, encoder_model, decoder_model


def conv_multi_model(n_input, n_output, n_feature, n_units, feat_units=5):
    root_word_input = Input(shape=(15, 28, 1), name="root_word_input")
    # word_index = Input(shape=(3,), name="word_index")
    feature_input = Input(shape=(n_feature,), name="word_feature_input")

    # feat = Concatenate(name="feature_word_index")([feature_input, word_index])
    feat_out = Dense(feat_units, activation="relu",
                     name="feature_output")(feature_input)

    x = Conv2D(20, (5, 5), padding='same', activation='relu',
               name="cnn")(root_word_input)
    x = MaxPooling2D(3, 3, name="pooling")(x)

    flat_output = Flatten(name="flatten")(x)
    x = Dense(n_units - feat_units, activation='relu',
              name="cnn_encoder")(flat_output)

    state_h = Concatenate(name="concatnate")([x, feat_out])

    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    decoder_gru = GRU(n_units, return_sequences=True,
                      return_state=True, name="decoder_gru")
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h)

    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([root_word_input, decoder_inputs,
                   feature_input], decoder_outputs)
    encoder_model = Model(
        [root_word_input, feature_input], state_h)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_outputs, state_h = decoder_gru(
        decoder_inputs, initial_state=decoder_state_input_h)

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input_h], [
                          decoder_outputs, state_h])

    return model, encoder_model, decoder_model


def onehot_model(n_words, n_output, n_feature, n_units, feat_units=5):
    root_word_input = Input(shape=(1,), name="root_word_input")
    feature_input = Input(shape=(n_feature,), name="word_feature_input")

    x = Embedding(n_words, (n_units - feat_units))(root_word_input)

    feat_out = Dense(feat_units, activation="relu",
                     name="feature_output")(feature_input)

    # x = Dense(n_units - feat_units, activation='relu', name="cnn_encoder")(flat_output)
    x = Flatten()(x)
    state_h = Concatenate(name="concatnate")([x, feat_out])

    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    decoder_gru = GRU(n_units, return_sequences=True,
                      return_state=True, name="decoder_gru")
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h)

    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([root_word_input, decoder_inputs,
                   feature_input], decoder_outputs)
    encoder_model = Model([root_word_input, feature_input], state_h)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_outputs, state_h = decoder_gru(
        decoder_inputs, initial_state=decoder_state_input_h)

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, decoder_state_input_h], [
                          decoder_outputs, state_h])

    return model, encoder_model, decoder_model

def rnn_multi_input(n_input, n_output, n_feature, n_enc_units, n_dec_units):
    # define training encoder
    feat_units = 15
    encoder_inputs = Input(shape=(None, n_input), name="root_word_input")
    encoder = LSTM(n_enc_units, return_state=True, name="encoder_lstm")
    # word_index = Input(shape=(3,), name="word_index")
    feature_input = Input(shape=(n_feature,), name="word_feature_input")
    # feat = Concatenate(name="feature_word_index")([feature_input, word_index])

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    
    feat_out = Dense(feat_units, activation="relu", name="feature_output")(feature_input)
    x = Concatenate()([state_h, feat_out])
    x2 = Concatenate()([state_c, feat_out])
    state_h = Dense(n_dec_units, activation='relu')(x)
    state_c = Dense(n_dec_units, activation='relu')(x2)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    decoder_lstm = LSTM(n_dec_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs, feature_input], decoder_outputs)

    encoder_model = Model([encoder_inputs, feature_input], encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_dec_units,))
    decoder_state_input_c = Input(shape=(n_dec_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
# def rnn_model_att(n_input, n_output, n_feature, n_enc_units, n_dec_units):
#     # define training encoder
#     feat_units = 15
#     encoder_inputs = Input(shape=(None, n_input), name="root_word_input")
#     encoder = LSTM(n_enc_units, return_state=True, name="encoder_lstm")
#     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
#     feature_input = Input(shape=(n_feature,), name="word_feature_input")
#     feat_out = Dense(feat_units, activation="relu", name="feature_output")(feature_input)
#     x = Concatenate()([state_h, feat_out])
#     x2 = Concatenate()([state_c, feat_out])
#     state_h = Dense(n_dec_units, activation='relu')(x)
#     state_c = Dense(n_dec_units, activation='relu')(x2)
#     encoder_states = [state_h, state_c]
    
#     decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
#     decoder_lstm = LSTM(n_dec_units, return_sequences=True, return_state=True, name="decoder_lstm")
#     decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

#     attn_layer = AttentionLayer(name='attention_layer')
#     attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
#     decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#     decoder_dense = Dense(n_output, activation='softmax', name="train_output")
#     decoder_outputs = decoder_dense(decoder_concat_input)
    
#     model = Model([encoder_inputs, decoder_inputs, feature_input], decoder_outputs)

#     encoder_model = Model([encoder_inputs, feature_input], encoder_states)
#     # define inference decoder
#     decoder_state_input_h = Input(shape=(n_dec_units,))
#     decoder_state_input_c = Input(shape=(n_dec_units,))
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#     decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]

#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

#     return model, encoder_model, decoder_model

def predict(infenc, infdec, inputs, n_steps, cardinality):
    # encode
    state = infenc.predict(inputs)
    # start of sequence input
    start = [0.0 for _ in range(cardinality)]
#     start[0] = 1
    target_seq = np.array(start).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h = infdec.predict([target_seq, state])
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = h
        # update target sequence
        target_seq = yhat
    return np.array(output)
