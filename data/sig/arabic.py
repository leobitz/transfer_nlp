import random
lines = open("arabic-task3-merged", encoding='utf-8').readlines()
data = []
for line in lines:
    root, feat, word = line[:-1].split('\t')
    data.append([root, feat, word])


random.shuffle(data)
print(len(data))
total = len(data)
file = open("arabic-test.txt", encoding='utf-8', mode='w')
for i in range(1600):
    root, feat, word = data[i]
    ft = feat.split(',')
    ft_new = []
    for pair in ft:
        left, right = pair.split('=')
        if "/" in right:
            right = right[1:-1].split('/')[0]
        f = "{0}={1}".format(left, right)
        ft_new.append(f)
    ft = ",".join(ft_new)
    line = "{0} {1} {2}\n".format(root, ft, word)
    file.write(line)
file.close()
file = open("arabic-train.txt", encoding='utf-8', mode='w')
for i in range(1600, len(data)):
    root, feat, word = data[i]
    ft = feat.split(',')
    ft_new = []
    for pair in ft:
        left, right = pair.split('=')
        if "/" in right:
            right = right[1:-1].split('/')[0]
        f = "{0}={1}".format(left, right)
        ft_new.append(f)
    ft = ",".join(ft_new)
    line = "{0} {1} {2}\n".format(root, ft, word)
    file.write(line)
file.close()
