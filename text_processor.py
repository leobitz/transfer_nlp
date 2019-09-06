import numpy as np
import json

# converts raw features to one-hot encoding

feat2val = json.loads(open('/home/leo/projects/transfer_nlp/features.json').read())
val2feat = {}
feat2int = {}
int2feat = {}
val2int = {}
char2int = {}
int2char = {}
i = 0
for featCat in feat2val:
    feat2int[featCat] = i
    int2feat[i] = featCat
    i += 1
i = 0
for featCat in feat2val:
    featVals = feat2val[featCat]
    for val in featVals:
        val2feat[val] = featCat
        val2int[val] = i
        i += 1
i = 0


def get_feat_val(val):
    featCat = val2feat[val]
    i = feat2val[featCat] .index(val)
    return i, featCat


rev_val2feat = {}
j = 0
for feat in feat2val.keys():
    val_list = feat2val[feat]
    max_len = len(val_list) + 1
    for i in range(len(val_list)):
        val = val_list[i]
        rev_val2feat[val] = [max_len, i]


def get_feature_array(data):
    """
    Creates a mapping from the words to thier features

    Params:
        data - raw text from the data file

    Returns:
        roots = root words list
        words = target words that are to be generated
        featureArray = the list of morphological features of the words
        maxroot = maximum length of a root word
        maxword = maximum length of a target word
    """
    lines = open(data, encoding='utf-8').readlines()
    n_featval = sum([len(feat2val[k]) for k in feat2val])
    featArray = np.zeros((len(lines), n_featval), dtype=np.int32)
    maxroot, maxword = 15, 0
    words = []
    roots = []
    word_index = []
    k = 0
    for line in lines:
        splited = line[:-1].split(' ')
#         if splited[3] == '1':
#             continue
        word = splited[0]
        root = splited[1]
        if len(splited) == 4:
            word_index.append([int(vs) for vs in splited[3]])
        wordFeats = [s[:-1] for s in splited[2].split('<')[2:]]
        words.append(word)
        roots.append(root)
        for feat in wordFeats:
            featArray[k][val2int[feat]] = 1
        k += 1
        if len(word) > maxword:
            maxword = len(word)
        # if len(root) > maxroot:
        #     maxroot = len(root)
        for char in word:
            if char not in char2int:
                temp = len(char2int)
                char2int[char] = temp
                int2char[temp] = char
    # start code
    int2char[len(char2int)] = '&'
    char2int['&'] = len(char2int)
    int2char[len(char2int)] = ' '
    char2int[' '] = len(char2int)
    for i, c in enumerate(sorted(char2int.keys())):
        char2int[c] = i
        int2char[i] = c
    if len(word_index) > 0:
        return roots, words,  featArray, word_index, maxroot, maxword
    return roots, words, featArray, maxroot, maxword
