import re
from nltk.corpus import stopwords
import nltk
import string
import shelve
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec
import itertools


# tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

######################################
###Clean data and use them as tokens##
######################################

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

# Define a function to split a review into parsed sentences
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


# plots word vectors
def plot_points(my_names, my_wv, dims=(1, 2), title = "None"):
    my_vectors = [my_vector_getter(elt, wv=my_wv) for elt in my_names]
    dim_1_coords = [element[0, dims[0] - 1] for element in my_vectors]
    dim_2_coords = [element[0, dims[1] - 1] for element in my_vectors]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.plot(dim_1_coords, dim_2_coords, 'ro')
    diff1 = max(dim_1_coords) - min(dim_1_coords)
    diff2 = max(dim_2_coords) - min(dim_2_coords)
    plt.axis([min(dim_1_coords) - 0.1 * diff1, max(dim_1_coords) + 0.1 * diff1, min(dim_2_coords) - 0.1 * diff2,
              max(dim_2_coords) + 0.1 * diff2])

    for x, y, name in zip(dim_1_coords, dim_2_coords, my_names):
        ax.annotate(name, xy=(x, y))

    plt.grid()
    plt.show()

def my_vector_getter(word, wv):
    try:
		# we use reshape because cosine similarity in sklearn now works only for multidimensional arrays
        word_array = wv[word].reshape(1,-1)
        return (word_array)
    except KeyError:
        print ('word: <', word, '> not in vocabulary!')

    def learn_word_representation(sentences, save=False):
        if rebuild:
            print("Learning model...")
            model = Word2Vec(sentences, size=100, min_count=5, workers=4)
            model.init_sims(replace=True)
            wordsVectors = {word: model[word] for word in model.vocab.keys()}
            if save:
                print("Saving model...")
                np.save('wordsVectors', wordsVectors)
        else:
            print("Loading word representations...")
            wordsVectors = np.load('wordsVectors.npy').item()
        return wordsVectors

    def build_vectors_for_mails(words_per_mail_id, vectors_for_words):
        vectors_for_mails = {}
        for mid, words in words_per_mail_id.iteritems():
            try:
                vectors_for_mails[mid] = np.mean(operator.itemgetter(*words)(vectors_for_words))
            except:
                vectors_for_mails[mid] = np.zeros(100)
                lgth = 0
                for word in words:
                    if word in vectors_for_words:
                        vectors_for_mails[mid] += vectors_for_words[word]
                        lgth += 1
                if lgth > 0:
                    vectors_for_mails[mid] /= lgth
                else:
                    del vectors_for_mails[mid]
        return vectors_for_mails

def eliminate_similar_words(dictionnary):
    l = len(dictionnary.keys())
    all_tokens = [token for sublist in dictionnary.values() for token in sublist]
    new_chain = []

    for i in all_tokens:
        new_chain = new_chain + i

    t_counts = dict(Counter(new_chain))

    tokens = [values for key, values in dictionnary.iteritems() if key < l]

    s = set(tokens)
    for key, values in dictionnary.iteritems():
        dictionnary[key] = [token for token in values if token in s]

    return dictionnary

################################################################################################################

'''
#train = []

#f = open("/Users/chen/Desktop/VCProjectDL/techcrunch_data2/1-7-million-people-are-testing-windows-10.txt")
#data = f.read()

#sentences = review_to_sentences(data,tokenizer)

text = np.load("/Users/chen/Desktop/VCProjectDL/all_tokens.npy")
sentences = text[()]
dictionnary = []

for sentence in sentences :
    w = min(len(sentence), 4)
    g = terms_to_graph(sentence,w)


    # decompose graph-of-words
    core_numbers = dict(zip(g.vs['name'], g.coreness()))
    # print "core_numbers", core_numbers

    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
    dictionnary.append(keywords)

np.save("all_keywords.npy", dictionnary)
'''

#function of embedding
# create empty word vectors for the words in vocabulary
# we set size=300 to match dim of GNews word vectors

def embedding_using_google_news(dictionnary, name):
    mcount = 10

    vectors = Word2Vec(size=3e2, min_count=mcount)


    vectors.build_vocab(dictionnary)

    vocab = [elt[0] for elt in vectors.wv.vocab.items()]

    all_tokens = [token for sublist in dictionnary for token in sublist]
#print(all_tokens)

    t_counts = dict(Counter(all_tokens))
    #print(t_counts)

# sanity check (should return True)
    len(vocab) == len([token for token, count in t_counts.iteritems() if count>=mcount])

# fill our word vectors
    path_to_wv = "/Users/chen/Desktop/VCProjectDL/code/Lib_Word2Vec/"

# we load only the Google word vectors corresponding to our vocabulary
    vectors.intersect_word2vec_format(path_to_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True)


    vectors_array = np.empty(shape=vectors.wv.syn0.shape)
    mapping = []

    for i, elt in enumerate(vectors.wv.vocab.items()):
        word = elt[0]
        mapping.append(word)
        vectors_array[i,] = vectors[word]

    my_pca = PCA(n_components=4)

    wv_2d_values = my_pca.fit_transform(vectors_array)
    print(wv_2d_values)

# finally, create dictionary with the words in vocab as keys and the 2-dimensional projections as values
    wv_2d = {}
    for i, word in enumerate(mapping):
        wv_2d[word] =  wv_2d_values[i,]

#print(wv_2d)
#print(wv_2d.values())
    #print(wv_2d.keys())
    words_subset = wv_2d.keys()
    plot_points(my_names=words_subset, my_wv=wv_2d, title = name)








dictionnary = np.load("/Users/chen/Desktop/Bi2VC/code/Lib_Word2Vec/100_tokens.npy")
dictionnary = dictionnary[()]
#print(dictionnary)

#embedding(dictionnary)




