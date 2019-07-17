<<<<<<< HEAD
import random

lines = open('words.txt').readlines()
random.shuffle(lines)
=======
import random

lines = open('words.txt').readlines()
random.shuffle(lines)
>>>>>>> 5bdcf056fb1926f8d3757ef8b0a4f25e990cc48d
open('rwords.txt', mode='w').writelines(lines)