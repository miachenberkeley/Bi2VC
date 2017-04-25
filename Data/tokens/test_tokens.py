import numpy as np
from collections import Counter
import xlsxwriter
import pandas as pd
import re


l_tokens = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/l_tokens.npy")
l_tokens = l_tokens[()]

print("l_tokens")
print(l_tokens[10:30])
print(len(l_tokens))


all_keywords_through_time = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/all_keywords_through_time.npy")
all_keywords_through_time = all_keywords_through_time[()]

print("all_keywords_through_time")
print([value for key, value in all_keywords_through_time.iteritems()])
print(type(all_keywords_through_time))
print(len(all_keywords_through_time))

all_keywords = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/all_keywords.npy")
all_keywords = all_keywords[()]

print("all_keywords")
print(all_keywords)
print(len(all_keywords))

all_tokens = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/all_tokens.npy")
all_tokens = all_tokens[()]

print("all_tokens")
print(all_tokens[:10])
print(len(all_tokens))