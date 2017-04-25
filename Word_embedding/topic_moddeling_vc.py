from gensim import corpora
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords

from library import clean_text_simple

# might also be required:
nltk.download('maxent_treebank_pos_tagger')
nltk.download('stopwords')

path = "/Users/chen/Desktop/DataProject/techcrunch data/"

file = open(path,"r")
review = file.read()
stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('english')

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())

with open(path_to_keywords + '\\' + filename, 'r') as my_file:
    text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text = re.sub('\s+', ' ', text)
    # convert to lower case
    text = text.lower()
    # turn string into list of keywords, preserving intra-word dashes
    # but breaking n-grams into unigrams to easily compute precision and recall
    keywords = text.split(';')
    keywords = [keyword.strip().split(' ') for keyword in keywords]
    # flatten list
    keywords = [keyword for sublist in keywords for keyword in sublist]
    # remove stopwords (rare but can happen due to n-gram breaking)
    keywords = [keyword for keyword in keywords if keyword not in stpwds]
    # apply Porter's stemmer
    keywords_stemmed = [stemmer.stem(keyword) for keyword in keywords]
    # remove duplicates (can happen due to n-gram breaking)
    keywords_stemmed_unique = list(set(keywords_stemmed))

    keywords_gold_standard.append(keywords_stemmed_unique)

from collections import defaultdict

frequency = defaultdict(int)

for text in texts:
         for token in text:
             frequency[token] +=1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

from pprint import pprint
pprint(texts)



