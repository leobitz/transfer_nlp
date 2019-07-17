<<<<<<< HEAD
import random
lines = open("wol-m.txt").readlines()

root_feat = {}
for line in lines:
    word, root, feat, i = line.split(' ')
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
        line = "{0} {1} {2} {3}\n".format(word, root, feat, i)
        lines.append(line)
# lines = sorted(lines)
random.shuffle(lines)

file = open("wol-aligned.txt", mode="w")
for line in lines:
    file.write(line)

file.close()
=======
import random
lines = open("wol-m.txt").readlines()

root_feat = {}
for line in lines:
    word, root, feat, i = line.split(' ')
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
        line = "{0} {1} {2} {3}\n".format(word, root, feat, v)
        lines.append(line)

random.shuffle(lines)

file = open("wol-aligned.txt", mode="w")
for line in lines:
    file.write(line)

file.close()
>>>>>>> 5bdcf056fb1926f8d3757ef8b0a4f25e990cc48d
