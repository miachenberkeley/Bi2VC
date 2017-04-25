import numpy as np
from collections import Counter
import xlsxwriter
import pandas as pd
import re


workbook = xlsxwriter.Workbook('tendance_per_month.xlsx')
worksheet = workbook.add_worksheet()

dictionnary = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/all_keywords_through_time.npy")
dictionnary = dictionnary[()]
#print(dictionnary.keys())

#print(dictionnary['2015-05-01'][0])

new_dictionnary = {}
colum = 0

for key_dict, value_dict in dictionnary.iteritems():
    worksheet.write(0, colum, key_dict)
    #print(key_dict)
    #print(len(value_dict))
    if len(value_dict) != 0:
        print(colum)
        key_words = {}


        all_tokens = [token for sublist in value_dict for token in sublist]

        t_counts = dict(Counter(all_tokens))
        print("len tokens %s" %len(all_tokens))
        print("t_counts %s" %len(t_counts))
        counter = 1
        for key, value in sorted(t_counts.iteritems(), key=lambda (k, v): (-v, k)):  # sorted by key
            worksheet.write(counter, colum, key)
            worksheet.write(counter, colum + 1, value)
            counter += 1
            key_words[key] = value
            #print("%s: %s" % (key, value))

        sort = sorted(key_words.iteritems(), key=lambda (k, v): (-v, k))

        print(len(sort), len(value_dict), len(all_tokens), len(t_counts))
        new_dictionnary[key_dict] = sort
        #print("done")

    colum += 2
    #print("colum %s" %colum)

np.save("key_words_sort.npy", new_dictionnary)


