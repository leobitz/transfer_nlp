import numpy as np
# from text_processor import *

class AEDataGen:

    def __init__(self):
        self.gwords = []
        self.wwords = []
        self.max_word_len = 0
        # gofa_lines = open('data/gof.txt').readlines()
        # wol_lines = open('data/wol.txt').readlines()
        lines = open('data/rwords.txt', encoding='utf-8').readlines()
        self.char2int = {}
        for i in range(len(lines)):
            # gline = gofa_lines[i][:-1].split(' ')
            # wline = wol_lines[i][:-1].split(' ')
            gword, wword = [w.strip().lower() for w in lines[i][:-1].split(' ')]
            self.gwords.append(gword)
            self.wwords.append(wword)
            if len(gword) > self.max_word_len:
                self.max_word_len = len(gword)
            if len(gword) > self.max_word_len:
                self.max_word_len = len(gword)
            for c in gword:
                self.char2int[c] = c
            for c in gword:
                self.char2int[c] = c
        self.int2char = {}
        j = 0
        self.char2int['&'] = '&'
        self.char2int[' '] = ' '
        for key in self.char2int.keys():
            self.int2char[j] = key
            self.char2int[key] = j
            j += 1
        self.n_chars = len(self.char2int)
        self.max_word_len += 2
        # print(len(self.char2int), self.max_word_len)

        
  
    def autoencoder_gen_data(self, batch_size=100, n_batches=-1, trainset=True):
        max_batch, min_batch = 0, 0
        if trainset == True:
            max_batch = int(len(self.wwords) * .7) // batch_size
            min_batch = 0
        else:
            max_batch = len(self.wwords)// batch_size
            min_batch = int(len(self.wwords) * .7 / batch_size)
        
        total_batchs = max_batch
        batch = min_batch
        # indexes = np.shuffle(np.arange(0, len(self.wwords)).astype(np.int32))
        while True:
            rootX, target_inX,  y =  list(), list(), list()
            for i in range(batch * batch_size, (1 + batch) * batch_size):
                root = self.wwords[i]
                word = self.gwords[i]
                root_encoded, target_encoded, target_in_encoded = self.encond_input_output(root, word)
                rootX.append(root_encoded.reshape((root_encoded.shape[0], root_encoded.shape[1], 1)))
                target_inX.append(target_in_encoded)
                y.append(target_encoded)
            yield [np.array(rootX), np.array(target_inX)], np.array(y)
            batch += 1
            if batch == total_batchs or batch == n_batches:
                batch = min_batch
                # indexes = np.shuffle(np.arange(0, len(self.wwords)).astype(np.int32))

    def word2vec(self, word, max_chars):
        vec = np.zeros((max_chars, self.n_chars))
        # print(word)
        for i in range(len(word)):
            vec[i][self.char2int[word[i]]] = 1
        index_sps = self.char2int[' ']
        vec[len(word):, index_sps] = 1
        return vec

    
    def encond_input_output(self, root_word, target_word):
        root_word = list(root_word)
        target_word = list(target_word) + ['&']
        target_word_in = ["&"] + target_word#[:-1]
        root_encoded = self.word2vec(root_word, self.max_word_len)
        target_encoded = self.word2vec(target_word, self.max_word_len)
        target_in_encoded = self.word2vec(target_word_in, self.max_word_len)
        return root_encoded, target_encoded, target_in_encoded

    def one_hot_decode(self, vec):
        return [self.int2char[np.argmax(v)] for v in vec]
    
    def word_sim(self, word1, word2):
        c = 0
        for i in range(len(word1)):
            if word1[i] == word2[i]:
                c += 1
        return c/len(word1)
            
