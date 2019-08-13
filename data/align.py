import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

lines = []
check = {}
man_lines = open('wol-data/manual.txt').readlines()
for line in man_lines:
    word, root, feat = line.split(' ')
    check[word] = word
    line = line[:-1] + " man"
    lines.append(line)




main_lines = open("wol-data/wolayitta_clean.txt").readlines()
for line in main_lines:
    line = line[:-1] + " norm"
    lines.append(line)

root_feat = {}
for line in lines:
    word, root, feat, ty = line.split(' ')
    suffix = word[len(root):]
    key = root + " " + feat
    if key not in root_feat:
        root_feat[key] = []
    root_feat[key].append(suffix + " " + ty)
    root_feat[key] = sorted(root_feat[key])

gen_lines = []
clean_lines = []
for key in root_feat:
    root, feat = key.split(' ')
    for i, suf in enumerate(root_feat[key]):
        suf, ty = suf.split(' ')
        word = root + suf
        v = ""
        if len(root_feat[key]) == 1:
            v = "100"
        else:
            if i == 0:
                v = "010"
            else:
                v = "001"
        # print(ty)
        if ty == "norm" and word not in check:
            gen_lines.append([word, root, feat, v])
        else:
            clean_lines.append([word, root, feat, v])

feat2val = json.loads(open('sig/features.json').read())

val2feat = {}
for key in feat2val.keys():
    val = feat2val[key]
    for k in val:
        val2feat[k] = key


def save(lines, name, shuffle=True):
    new_lines = []
    for line in lines:
        word, root, feat, index = line
        wordFeats = [s[:-1] for s in feat.split('<')[2:]]
        feat = []
        for f in wordFeats:
            feat.append("{0}={1}".format(val2feat[f], f))
        feat.append("gen=" + index)
        line = "{0} {1} {2}\n".format(root, ",".join(feat), word)
        new_lines.append(line)

    new_lines = list(set(new_lines))
    if shuffle:
        random.shuffle(new_lines)
    open(name, mode="w").writelines(new_lines)


save(gen_lines[:int(.7 * len(gen_lines))], "wol/wol-train.txt", shuffle=True)
save(gen_lines[int(.7 * len(gen_lines)):], "wol/wol-test.txt", shuffle=True)
save(clean_lines, "wol/wol-clean-test.txt", shuffle=True)
