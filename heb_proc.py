import numpy as np

def read(file='data/heb/hebrew-task1-train', feature_map=None):
    lines = open(file, encoding='utf-8').readlines()
    feats = {}
    root2word = {}
    root2feat = {}
    for line in lines:
        line = line[:-1]
        root, feat, word = line.split('\t')
        
        feat = feat.split(',')
        current_feat = {}
        for f in feat:
            key, val = f.split('=')
            current_feat[key] = val
        root2feat[root] = current_feat
        root2word[root] = word
        
        for key in current_feat.keys():
            if key not in feats:
                feats[key] =[]
            val = current_feat[key]
            if val not in feats[key]:
                feats[key].append(val)
    
    if feature_map is not None:
        feats = feature_map
    for root in root2feat.keys():
        root2feat[root] = feat2vec(feats,root2feat[root])

    return feats, root2feat, root2word


def feat2vec(feat_map, word_feat):
    vec = []
    for key in feat_map:
        if key in word_feat:
            val = word_feat[key]
            feat_length = len(feat_map[key])
            feat_index = feat_map[key].index(val)
            feat_vec = [0]*(feat_length + 1)
            feat_vec[feat_index] = 1
            vec.extend(feat_vec)
        else:
            feat_length = len(feat_map[key])
            feat_vec = [0]*(feat_length + 1)
            feat_vec[-1] = 1
            vec.extend(feat_vec)
    return np.array(vec)

def word2vec(char2int, word, max_chars):
    vec = np.zeros((max_chars, len(char2int)))
    for i in range(len(word)):
        vec[i][char2int[word[i]]] = 1
    index_sps = char2int[' ']
    vec[len(word):, index_sps] = 1
    return vec       


# def read(file='data/heb/hebrew-task1-train'):
#     lines = open(file, encoding='utf-8').readlines()
#     chars = {}
#     feats = {}
#     root2word = {}
#     root2feat = {}
#     max_root_length = 0
#     max_word_length = 0
#     for line in lines:
#         line = line[:-1]
#         root, feat, word = line.split('\t')
#         if len(root) > max_root_length:
#             max_root_length = len(root)
#         if len(word) > max_word_length:
#             max_word_length = len(word)
        
#         feat = feat.split(',')
#         current_feat = {}
#         for f in feat:
#             key, val = f.split('=')
#             current_feat[key] = val
#         root2feat[root] = current_feat
#         root2word[root] = word
#         for key in current_feat.keys():
#             if key not in feats:
#                 feats[key] =[]
#             val = current_feat[key]
#             if val not in feats[key]:
#                 feats[key].append(val)
        
            
#         for c in root:
#             chars[c] = c
#         chars['&'] = '&'
#         chars[' '] = ' '
#         i = 0
#         for key in chars.keys():
#             chars[key] = i
#             i += 1
#     for root in root2feat.keys():
#         root2feat[root] = feat2vec(feats,root2feat[root])
#     feat_length = len(root2feat[root])
#     return chars, feats, root2feat, root2word, max_root_length, max_word_length, feat_length


def stat(root2feat, root2word, features):
    char2int = {}
    max_word = 0
    max_root = 0
    for root in root2feat:
        word = root2word[root]
        if len(root) > max_root:
            max_root = len(root)
        for c in root:
            if c not in char2int:
                char2int[c] = len(char2int)
        if len(word) > max_word:
            max_word = len(word)
        for c in root2word[root]:
            if c not in char2int:
                char2int[c] = len(char2int)
    char2int['&'] = len(char2int)
    char2int[' '] = len(char2int)
    feat_length = len(root2feat[root])
    int2char = {val: key for val, key in enumerate(char2int)}
    return char2int, int2char, max_root, max_word, feat_length
    
            



def get_data(root2word, root2feat, char2int, max_root, max_word, feat_length):
    roots_mat = np.zeros((len(root2feat), max_root, len(char2int)))
    out_words_mat = np.zeros((len(root2feat), max_word+2, len(char2int)))
    decoder_words_mat = np.zeros((len(root2feat), max_word+2, len(char2int)))
    feats_vec = np.zeros((len(root2feat), feat_length))
    i = 0
    for root in root2feat.keys():
        vec = root2feat[root]
        word = root2word[root] + "&"
        decode_word = "&" + word
        root_mat = word2vec(char2int, root, max_root)
        word_mat = word2vec(char2int, word, max_word + 2)
        decoder_mat = word2vec(char2int, decode_word, max_word + 2)
        roots_mat[i] = root_mat
        feats_vec[i] = vec
        decoder_words_mat[i] = decoder_mat
        out_words_mat[i] = word_mat
        i += 1
    return roots_mat, decoder_words_mat, feats_vec, out_words_mat

def gen_data(roots_mat, decode_mat, feats_vec,words_mat, batch_size=100):
    batch = 0
    indexes = np.arange(len(roots_mat), dtype=np.int32)
    while True:
        batch_indexes = indexes[batch: batch + batch_size]
        roots = roots_mat[batch_indexes].reshape((batch_size, roots_mat.shape[1], roots_mat.shape[2], 1))
        words = words_mat[batch_indexes]
        feats = feats_vec[batch_indexes]
        decode = decode_mat[batch_indexes]
        batch += batch_size
        if batch + batch_size > len(indexes):
            batch = 0
            np.random.shuffle(indexes)
        yield [roots, feats, decode], words

def one_hot_decode(vec, int2char):
    return [int2char[np.argmax(v)] for v in vec]
