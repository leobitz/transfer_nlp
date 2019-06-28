gofa_lines = open("goffa.txt").readlines()
wol_lines = open("wolayitta_clean.txt").readlines()

gofa = {}
for line in gofa_lines:
    word, root, feat = line[:-1].split(" ")
    gofa[word.strip()] = [root, feat]

wol = {}
for line in wol_lines:
    word, root, feat = line[:-1].split(" ")
    wol[word.strip()] = [root, feat]

wordword = open("word-word").readlines()

dic = {}
for ww in wordword:
    ww = ww[:-1].split(' ')
    dic[ww[1]] = ww[0]

valid = []
for key in gofa.keys():
    if key in dic:
        w = dic[key]
        if w in wol.keys():
            valid.append([w, key])

wfile= open("gof.txt", mode='w')
for i in range(len(valid)):
    w, g = valid[i]
    r, f = gofa[g]
    line = "{0} {1} {2}\n".format(w, r, f)
    wfile.write(line)

wfile.close()