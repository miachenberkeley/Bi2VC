from library import clean_text_simple, terms_to_graph
from nltk.corpus import stopwords
from collections import Counter

import xlsxwriter
from os import listdir
import numpy as np



path_to_data = "/Users/chen/Desktop/Bi2VC/techcrunch data/"

stpwds = stopwords.words('english')

#extract key words
key_words_gow = {}
counter = 0


# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('keywords_description_tokens.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"name of file")
worksheet.write(0,1,"key words")



dictionnary = []




##########
# gow #
##########

for f in listdir('/Users/chen/Desktop/Bi2VC/techcrunch data/'):

    worksheet.write(counter+1, 0, f)

    file = open('/Users/chen/Desktop/Bi2VC/techcrunch data/' + f, "r")

    review = file.read()
    # pre-process document
    try :
        my_tokens = clean_text_simple(review)
    except :
        print "info", review
        #print "my_tokens", my_tokens
        #worksheet.write_row(counter + 1, 1,  my_tokens)
        #worksheet.write(counter + 1, 2,  "il y a une erreur")


    if len(my_tokens) == 0:
        pass
    elif len(my_tokens) == 1:
        keywords = my_tokens
        #worksheet.write(counter + 1, 1,  keywords)

    else :

        w = min(len(my_tokens),4)
        #print "w", w
        g = terms_to_graph(my_tokens, w)

        # decompose graph-of-words
        core_numbers = dict(zip(g.vs['name'], g.coreness()))
        #print "core_numbers", core_numbers

        max_c_n = max(core_numbers.values())
        keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
        print(keywords)
        #worksheet.write_row(counter + 1, 1, keywords)
        # save results

    dictionnary.append(keywords)



    counter += 1
    if counter % 100 == 0:
        print counter, 'body processed'


np.save("keywords_description_tokens.npy", dictionnary)


text = np.load("all_tokens.npy")
dictionnary = text[()]
dictionnary_sort = {}
all_tokens = [token for sublist in dictionnary for token in sublist]
#print(all_tokens)

t_counts = dict(Counter(all_tokens))

for key, value in sorted(t_counts.iteritems(), key=lambda (k,v): (v,k)):
    dictionnary_sort[value] = key
    print("%s: %s" % (key, value))

print(dictionnary_sort)