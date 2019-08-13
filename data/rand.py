main_lines = open('wol-data/wolayitta_clean.txt').readlines()
manul_lines = open('wol-data/manual.txt').readlines()

d = {}
for line in manul_lines:
    word, root, feat = line[:-1].split()
    d[word] = root

k = 0
for ml in main_lines:
    word, root, feat= ml[:-1].split()
    if word in d:
        k += 1

print(k, len(manul_lines))
