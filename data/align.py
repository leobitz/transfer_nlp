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
