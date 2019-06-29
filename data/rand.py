import random

lines = open('words.txt').readlines()
random.shuffle(lines)
open('rwords.txt', mode='w').writelines(lines)