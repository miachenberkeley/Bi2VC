import numpy as np
import collections
from collections import OrderedDict
from collections import Counter
import xlsxwriter
import pandas as pd
import re
'''

#workbook = xlsxwriter.Workbook('tendance_300_per_month.xlsx')
#worksheet = workbook.add_worksheet()

A = pd.read_excel("/Users/chen/Desktop/Bi2VC/data/keywords_trends.xlsx")
list_words = A['vr']

dictionnary = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/all_keywords_through_time.npy")
dictionnary = dictionnary[()]
#allows to read column

new_dictionnary = {}
colum = 0
for key_dict, value_dict in dictionnary.iteritems():
    #worksheet.write(0, colum, key_dict)

    key_words = {}
    if len(value_dict) != 0:

        all_tokens = [token for sublist in value_dict for token in sublist]
        t_counts = dict(Counter(all_tokens))
        counter = 1
        for key, value in sorted(t_counts.iteritems(), key=lambda (k, v): (-v, k)):  # sorted by key
            #print(key,list_words )
            for i in list_words:
                if i == key :
                    counter += 1
                    key_words[key] = value
                    #worksheet.write(counter, colum, key)
                    #worksheet.write(counter, colum + 1, value)

                else:
                    pass



    colum += 2
    #new_dictionnary[key_dict] = sorted(key_words.iteritems(), key = lambda (k,v):(-v,k))
    new_dictionnary[key_dict] = key_words

#print(new_dictionnary)
#np.save("key_words_sort_300_no_order.npy", new_dictionnary) '''

workbook = xlsxwriter.Workbook('tokens.xlsx')
worksheet = workbook.add_worksheet()

new_dictionnary = np.load("key_words_sort_300_no_order.npy")[()]
kw_throughtime = {}

#sorted dictionnary by time order

print(sorted(new_dictionnary.keys()))

new = collections.OrderedDict(sorted(new_dictionnary.items()))

###########################


for keys, values in new.iteritems():
    print("keys", keys)
    #print("values", values)
    #print(values.keys())
    for word in values.keys():
        #print("key",kw_throughtime.keys())
        if word in kw_throughtime.keys():
            kw_throughtime[word].append([keys, values[word]])
        else:
            kw_throughtime[word] = []


print(kw_throughtime.keys())
for counter, value in enumerate (kw_throughtime.keys()):
    worksheet.write(counter, 0, value)

#np.save("key_words_frequency_throughtime.npy",kw_throughtime)
