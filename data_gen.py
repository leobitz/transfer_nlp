import numpy as np
from text_processor import *


class DataGen:

    def __init__(self, reverse=False, data=None):
        if reverse:
            self.roots, self.words, self.featArray, mr, mw, n_features = get_reverse_feature_array(
                data)
            self.n_features = n_features
            self.pred_feat_shapes = []
            for feat in self.featArray[0]:
                self.pred_feat_shapes.append(len(feat))
        else:
            parsed = get_feature_array(data)
            print(len(parsed))
            if len(parsed) == 5:
                self.roots, self.words, self.featArray, mr, mw = parsed
            else:
                self.roots, self.words, self.featArray, self.word_indexes, mr, mw = parsed
            self.word_feat_len = len(self.featArray[0])
        vocab = list(set(self.roots))
        self.root2int = {key: val for val, key in enumerate(vocab)}
        self.int2word = {
            self.root2int[key]: key for key in list(self.root2int.keys())}
        self.n_chars = len(char2int)
        self.max_root_len = mr
        self.max_output_len = mw + 2

    def onehot_gen_data(self, batch_size=100, n_batches=-1, trainset=True):
        max_batch, min_batch = 0, 0
        if trainset == True:
            max_batch = int(len(self.words) * .7) / batch_size
            min_batch = 0
        else:
            max_batch = len(self.words) / batch_size
            min_batch = int(len(self.words) * .7 / batch_size)

        total_batchs = max_batch
        batch = min_batch
        while True:
            rootX, target_inX, featX, y = list(), list(), list(), list()
            for i in range(batch * batch_size, (1 + batch) * batch_size):
                root = self.roots[i]
                word = self.words[i]
                word_feature = self.featArray[i]
                root_encoded, target_encoded, target_in_encoded = self.encond_input_output(
                    root, word)
                rootX.append(self.root2int[root])
                target_inX.append(target_in_encoded)
                featX.append(word_feature)
                y.append(target_encoded)
            yield [np.array(rootX), np.array(target_inX), np.array(featX)], np.array(y)
            batch += 1
            if batch == total_batchs or batch == n_batches:
                batch = min_batch

    def cnn_gen_data(self, batch_size=100, n_batches=-1, trainset=True):
        max_batch, min_batch = 0, 0
        if trainset == True:
            max_batch = int(len(self.words) * .7) / batch_size
            min_batch = 0
        else:
            max_batch = len(self.words) / batch_size
            min_batch = int(len(self.words) * .7 / batch_size)

        total_batchs = max_batch
        batch = min_batch
        while True:
            rootX, target_inX, featX, y = list(), list(), list(), list()
            for i in range(batch * batch_size, (1 + batch) * batch_size):
                root = self.roots[i]
                word = self.words[i]
                word_feature = self.featArray[i]
                root_encoded, target_encoded, target_in_encoded = self.encond_input_output(
                    root, word)
                rootX.append(root_encoded.reshape(
                    (root_encoded.shape[0], root_encoded.shape[1], 1)))
                target_inX.append(target_in_encoded)
                featX.append(word_feature)
                y.append(target_encoded)
            yield [np.array(rootX), np.array(target_inX), np.array(featX)], np.array(y)
            batch += 1
            if batch == total_batchs or batch == n_batches:
                batch = min_batch

    def cnn_gen_data_multi_word(self, batch_size=100, n_batches=-1, trainset=True):
        max_batch, min_batch = 0, 0
        if trainset == True:
            max_batch = int(len(self.words) * .7) / batch_size
            min_batch = 0
        else:
            max_batch = len(self.words) / batch_size
            min_batch = int(len(self.words) * .7 / batch_size)

        total_batchs = max_batch
        batch = min_batch
        while True:
            rootX, target_inX, featX, y = list(), list(), list(), list()
            word_indexes = []
            for i in range(batch * batch_size, (1 + batch) * batch_size):
                root = self.roots[i]
                word = self.words[i]

                # word_index = [0, 0]
                # word_index[self.word_indexes[i]] = 1
                word_feature = self.featArray[i]

                root_encoded, target_encoded, target_in_encoded = self.encond_input_output(
                    root, word)
                rootX.append(root_encoded.reshape(
                    (root_encoded.shape[0], root_encoded.shape[1], 1)))

                target_inX.append(target_in_encoded)
                featX.append(word_feature)
                y.append(target_encoded)
                word_indexes.append(self.word_indexes[i])
            feat = np.concatenate([np.array(featX), np.array(word_indexes)], axis=1)
            yield [np.array(rootX), np.array(target_inX), feat], np.array(y)
            batch += 1
            if batch == total_batchs or batch == n_batches:
                batch = min_batch

    def rnn_gen_data(self, batch_size=100, n_batches=-1, trainset=True):
        max_batch, min_batch = 0, 0
        if trainset == True:
            max_batch = int(len(self.words) * .7) / batch_size
            min_batch = 0
        else:
            max_batch = len(self.words) / batch_size
            min_batch = int(len(self.words) * .7 / batch_size)

        total_batchs = max_batch
        batch = min_batch
        while True:
            rootX, target_inX, featX, y = list(), list(), list(), list()
            word_indexes = []
            for i in range(batch * batch_size, (1 + batch) * batch_size):
                root = self.roots[i]
                word = self.words[i]
                word_feature = self.featArray[i]
                root_encoded, target_encoded, target_in_encoded = self.encond_input_output(
                    root, word)
                rootX.append(root_encoded.reshape(
                    (root_encoded.shape[0], root_encoded.shape[1])))
                target_inX.append(target_in_encoded)
                featX.append(word_feature)
                y.append(target_encoded)
                word_indexes.append(self.word_indexes[i])
            feat = np.concatenate([np.array(featX, dtype=np.float32), np.array(word_indexes, dtype=np.float32)], axis=1)
            yield [np.array(rootX, dtype=np.float32), np.array(target_inX, dtype=np.float32), feat], np.array(y)
            batch += 1
            if batch == total_batchs or batch == n_batches:
                batch = min_batch

    def word2vec(self, word, max_chars):
        vec = np.zeros((max_chars, self.n_chars))
        for i in range(len(word)):
            vec[i][char2int[word[i]]] = 1
        index_sps = char2int[' ']
        vec[len(word):, index_sps] = 1
        return vec

    def encond_input_output(self, root_word, target_word):
        root_word = list(root_word)
        target_word = list(target_word) + ['&']
        target_word_in = ["&"] + target_word  # [:-1]
        root_encoded = self.word2vec(root_word, self.max_root_len)
        target_encoded = self.word2vec(target_word, self.max_output_len)
        target_in_encoded = self.word2vec(target_word_in, self.max_output_len)
        return root_encoded, target_encoded, target_in_encoded

    def one_hot_decode(self, vec):
        return [int2char[np.argmax(v)] for v in vec]

    def word_sim(self, word1, word2):
        c = 0
        for i in range(len(word1)):
            if word1[i] == word2[i]:
                c += 1
        return c/len(word1)
