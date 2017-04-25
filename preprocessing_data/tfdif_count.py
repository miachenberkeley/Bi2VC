from library import clean_text_simple, terms_to_graph
from nltk.corpus import stopwords
from collections import Counter

import xlsxwriter
from os import listdir
import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
stpwds = stopwords.words('english')

'''
### path to data ####

path_to_data = "/Users/chen/Desktop/Bi2VC/Data/investor_data.xlsx"

df = pandas.read_excel(path_to_data)
descriptions = df['investor properties - description'].values



### extract words and make them as tokens ###


description_cleaned = []
counter = 0
RightCounter = 0
for review in descriptions:
    try:

        my_tokens = clean_text_simple(review)
        #print("yes", my_tokens)
        description_cleaned.append(my_tokens)
        RightCounter +=1
        if RightCounter%100 == 0:
            print("100 reviews processings")
    except:
        counter +=1
        if counter%100 == 0:
            print("100 text does not work")

        #print ("info", review)

print(len(description_cleaned))
#save clean tokens#

#np.save("cleaned_startup_description_tokens.npy",description_cleaned )



abstracts_cleaned_strings = [' '.join(elt) for elt in description_cleaned]

tfidf_vectorizer = TfidfVectorizer(stop_words=stpwds)
doc_term_matrix = tfidf_vectorizer.fit_transform(abstracts_cleaned_strings)
terms = tfidf_vectorizer.get_feature_names()
vectors_list = doc_term_matrix.todense().tolist()

keywords_tfidf = []
counter = 0

for vector in vectors_list:

        # bow feature vector as list of tuples
        terms_weights = zip(terms, vector)
        # keep only non zero values (the words in the document)
        nonzero = [tuple for tuple in terms_weights if tuple[1] != 0]
        # rank by decreasing weights
        nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
        # retain top 33% words as keywords
        numb_to_retain = int(round(len(nonzero) / 3))
        keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]

        keywords_tfidf.append(keywords)

        counter += 1
        if counter % 100 == 0:
            print (counter, 'vectors processed')

#save key_words#
#np.save("cleaned_startup_tfidf_kw.npy",keywords_tfidf ) '''

######### put it in a worksheet ##########@


# Create a workbook and add a worksheet.
text = np.load("cleaned_startup_tfidf_kw.npy")
dictionnary = text[()]

workbook = xlsxwriter.Workbook('keywords_description_tokens.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"key words")
worksheet.write(0,1,"frequency of the word")



#count the frequency of the world

dictionnary_sort = {}
all_tokens = [token for sublist in dictionnary for token in sublist]
#print(all_tokens)

t_counts = dict(Counter(all_tokens))
counter = 0
for key, value in sorted(t_counts.iteritems(), key=lambda (k,v): (v,k)):

    worksheet.write(counter, 0, key)
    worksheet.write(counter, 1, value)
    counter += 1
    print("%s: %s" % (key, value))

