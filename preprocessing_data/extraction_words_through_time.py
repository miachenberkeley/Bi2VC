import pandas as pd
import json
import numpy as np
import string
import re

from nltk import pos_tag

import nltk
from nltk.corpus import stopwords
nltk.download('maxent_treebank_pos_tagger')
nltk.download('stopwords')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def time_to_num(time_str):
    yy, mm, dd = map(int, time_str.split('-'))
    return yy * 365 + mm * 12 + dd

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

punct = string.punctuation.replace('-', '')
stpwds = stopwords.words('english')

def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True):
    for i in punct:
        text = text.replace(i,' ')
    # convert to lower case
    text = text.lower()


    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)



    #remove no letters
    text = re.sub("[^a-zA-Z]"," ", text)

    # strip extra white space
    # strip leading and trailing white space
    # tokenize (split based on whitespace)
    tokens = text.split()
    if pos_filtering == True:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return(tokens)


#data

year = ['2015-01-01','2015-02-01', '2015-03-01', '2015-04-01', '2015-05-01', '2015-06-01',
       '2015-07-01','2015-08-01','2015-09-01','2015-10-01','2015-11-01','2015-12-01',
        '2016-01-01',
        '2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01']

times = {}


with open('/Users/chen/Desktop/Bi2VC/code/metadata/metadata_full.json') as files:
    metadata = json.load(files)

df = pd.DataFrame(metadata)
df = df.transpose()  #change rows and columns
df = df.reset_index()

df['url'] = df['index']
del df['index']

texts = []
titles = df['filename']
for title in titles:
    path = '/Users/chen/Desktop/Bi2VC/techcrunchData/' + title
    with open(path) as data:
        texts.append(data.read())
df['content'] = pd.Series(texts)
#df.head()[0:1]

df['timestamp'] = df['timestamp'].apply(lambda x: str(x).split()[0])


for ynumber, i in enumerate(year):
    tokens = []
    try:

        print(i, year[ynumber + 1], "1st loop")
    except:

        print(len(year), i, ynumber)
    for number, time in enumerate(df['timestamp']):
        a = time_to_num(time)
        try:
            #print(time_to_num(i), a, time_to_num(year[ynumber + 1]), "2nd loop")
            if (time_to_num(i) <= a) and (a <= time_to_num(year[ynumber + 1])):
                kw = clean_text_simple(df['content'][number], stpwds)
                #tokens.append(time)
                #print(tokens,"tokens")
                tokens.append(kw)

        except:
            #print(i,time,ynumber + 1 )
            pass
    print(i)
    times[i] = tokens

np.save("all_keywords_through_time.npy", times)

for i in year:
 print(len(times[i]))
 print("hello")







