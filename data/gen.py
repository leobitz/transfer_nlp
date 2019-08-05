# gofa_lines = open("goffa.txt").readlines()
# wol_lines = open("wolayitta_clean.txt").readlines()

# gofa = {}
# for line in gofa_lines:
#     word, root, feat = line[:-1].split(" ")
#     gofa[word.strip()] = [root, feat]

# wol = {}
# for line in wol_lines:
#     word, root, feat = line[:-1].split(" ")
#     wol[word.strip()] = [root, feat]

# wordword = open("word-word").readlines()

# dic = {}
# for ww in wordword:
#     ww = ww[:-1].split(' ')
#     dic[ww[1]] = ww[0]

# valid = []
# for key in gofa.keys():
#     if key in dic:
#         w = dic[key]
#         if w in wol.keys():
#             valid.append([w, key])

# wfile= open("gof.txt", mode='w')
# for i in range(len(valid)):
#     w, g = valid[i]
#     r, f = gofa[g]
#     line = "{0} {1} {2}\n".format(w, r, f)
#     wfile.write(line)

# wfile.close()
lines = open('wolaytta-train.txt', encoding='utf-8').readlines()
open('woll-14-train.txt', mode='w', encoding='utf-8').writelines(lines[:14780])
open('woll-29-train.txt', mode='w', encoding='utf-8').writelines(lines[:14780*2])
open('woll-44-train.txt', mode='w', encoding='utf-8').writelines(lines[:14780*3])

open('woll-14-test.txt', mode='w', encoding='utf-8').writelines(lines[-3500:])
open('woll-29-test.txt', mode='w', encoding='utf-8').writelines(lines[-7000:])
open('woll-44-test.txt', mode='w', encoding='utf-8').writelines(lines[-14000:])

