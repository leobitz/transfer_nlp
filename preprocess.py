
import numpy as np
def process():
    chars = {}
    feats = {}
    max_root, max_word = 0, 0
    langs = ["arabic", "finnish", "georgian", "german", "hungarian", "navajo", "russian", "spanish", "turkish"]

    for lang in langs:
        lines = open("data/sig/{0}-train.txt".format(lang), encoding='utf-8').readlines()
        for line in lines:
            root, feat, word = line[:-1].split(' ')
            feat = feat.split(',')
            for ft in feat:
                key, val = ft.split('=')
                if key not in feats:
                    feats[key] = []

                if val not in feats[key]:
                    feats[key].append(val)
            if len(root) > max_root:
                max_root = len(root)
            if len(word) > max_word:
                max_word = len(word)

            for c in word:
                if c not in chars:
                    chars[c] = len(chars)
            for c in root:
                if c not in chars:
                    chars[c] = len(chars)

    chars[' '] = len(chars)
    chars['<'] = len(chars)
    chars['>'] = len(chars)
    return chars, feats, max_root, max_word

def word_to_index(word, char2int, max_len, pad_char):
    vec = [char2int[w] for w in word]
    vec = vec + [char2int[pad_char]]*(max_len - len(word))
    return vec

def index_to_word(indexes, int2char):
    chars = []
    for i in indexes:
        chars.append(int2char[i])
    return chars


def feat_to_vec(feat, val, feat2val):
    feat_index = feat2val[feat].index(val)
    vec = [0] * (len(feat2val[feat]) + 1)
    vec[feat_index] = 1
    return vec

def word_index_2_one_hot(indexes, max_len):
    vecs = []
    for i in indexes:
        vec = [0] * max_len
        vec[i] = 1
        vecs.append(vec)
    return vecs
    

def convert(char2int, feat2val, max_root_len, max_word_len, train_set=True):
    max_root_len = max_root_len + 2
    max_word_len = max_word_len + 2
    langs = ["arabic", "finnish", "georgian", "german", "hungarian", "navajo", "russian", "spanish", "turkish"]
    root_data, feat_data, in_data, out_data = [], [] ,[], []
    for lang in langs:
        if train_set == True:
            name = "data/sig/{0}-train.txt".format(lang)
        else:
            name = "data/sig/{0}-test.txt".format(lang)
        lines = open(name, encoding='utf-8').readlines()
        for line in lines:
            root, feat, word = line[:-1].split(' ')
            feat = feat.split(',')
            current_feats = {}
            feat_vecs = []
            for ft in feat:
                key, val = ft.split('=')
                current_feats[key] = val

            for ft in feat2val:
                if ft in current_feats:
                    feat_vecs += feat_to_vec(ft, current_feats[ft], feat2val)
                else:
                    vec = [0]*(len(feat2val[ft]) + 1)
                    vec[-1] = 1
                    feat_vecs += vec

            root_vec = word_to_index('<' + root + '>', char2int, max_root_len, ' ')

            word_vec_in = word_to_index('<' + word + '>', char2int, max_word_len, ' ')
            word_vec_out = word_to_index(word + "> ", char2int, max_word_len, ' ')
            output = word_index_2_one_hot(word_vec_out, len(char2int))

            root_vec = np.array(root_vec, dtype=np.int32)
            feat_vecs = np.array(feat_vecs, dtype=np.float32)
            word_vec_in = np.array(word_vec_in, dtype=np.int32)
            output = np.array(output, dtype=np.float32)

            root_data.append(root_vec)
            feat_data.append(feat_vecs)
            in_data.append(word_vec_in)
            out_data.append(output)
    root_data = np.stack(root_data)
    feat_data = np.stack(feat_data)
    in_data = np.stack(in_data)
    out_data = np.stack(out_data)
    return root_data, feat_data, in_data, out_data

def gen(data, batch_size=64, max_batch=-1):
    current = 0
    indexes = list(range(len(data[0])))
    np.random.shuffle(indexes)
    while True:
        batch_indexes = indexes[current: current + batch_size]
        batch_root = data[0][batch_indexes]
        batch_feat = data[1][batch_indexes]
        batch_in = data[2][batch_indexes]
        batch_out = data[3][batch_indexes]
        current += batch_size

        if current > len(data[0]):
            np.random.shuffle(indexes)
            current = 0

        yield [batch_root, batch_feat, batch_in], batch_out
        

# print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
# print(word_to_index("leo", char2int, 6, ' '))
# print(feat_to_vec('poss', 'PSS2P', feat2val))
# print(feat2val['poss'])
# k = 0
# for key in feat2val.keys():
#     print(key, feat2val[key])
#     k += len(feat2val[key])
