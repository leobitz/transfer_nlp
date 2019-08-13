import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

man_lines = open('wol-data/manual.txt').readlines()
man_lines = [line[:-1].split() for line in man_lines]
not_words = {line[0]: line[0] for line in man_lines}
print(man_lines[0])
lines = open(args.input).readlines()

root_feat = {}
for line in lines:
    word, root, feat = line[:-1].split(' ')
    suffix = word[len(root):]
    key = root + " " + feat
    if key not in root_feat:
        root_feat[key] = []
    root_feat[key].append(suffix)
    root_feat[key] = sorted(root_feat[key])

lines = []
for key in root_feat:
    root, feat = key.split(' ')
    for i, suf in enumerate(root_feat[key]):
        word = root + suf
        v = ""
        if len(root_feat[key]) == 1:
            v = "100"
        else:
            if i == 0:
                v = "010"
            else:
                v = "001"
        # line = "{0} {1} {2} {3}".format(word, root, feat, v)
        lines.append([word, root, feat, v])
# print(lines[0])

feat2val = json.loads(open('sig/features.json').read())

# print(feat2val)
val2feat = {}
for key in feat2val.keys():
    val = feat2val[key]
    for k in val:
        val2feat[k] = key

new_lines = []
for line in lines:
    word, root, feat, index = line
    if word in not_words:
        continue
    wordFeats = [s[:-1] for s in feat.split('<')[2:]]
    feat = []
    for f in wordFeats:
        feat.append("{0}={1}".format(val2feat[f], f))
    feat.append("gen=" + index)
    line = "{0} {1} {2}\n".format(root, ",".join(feat), word)
    new_lines.append(line)

lines = list(set(new_lines))
random.shuffle(lines)

trains = lines[:int(0.7 * len(lines))]
tests = lines[int(0.7 * len(lines)):]

open(args.output + "-train.txt", mode="w").writelines(trains)
open(args.output + "-test.txt", mode="w").writelines(tests)