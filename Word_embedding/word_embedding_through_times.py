from wordEmbedding_test2 import embedding_using_google_news, eliminate_similar_words
import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
import gensim
import re


def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

###################@ Train ###################
##############################################
documents = np.load("/Users/chen/Desktop/Bi2VC/code/Lib_Word2Vec/l_tokens.npy")
documents = documents[()]

#print(documents)

#########################################################################################
################Load########################################
################Clustering###################################
########################################

dictionnary = np.load("/Users/chen/Desktop/Bi2VC/code/Keyword_extraction/all_keywords_through_time.npy")
dictionnary = dictionnary[()]

print(dictionnary)

#print(dictionnary)
'''
for key, value in dictionnary.items():
    if len(value) != 0:
        all_tokens = [token for sublist in value for token in sublist]
        tampon = dict(Counter(all_tokens))
        tokens =[key for key, value in tampon.items() if value >10 ]

        model.build_vocab(tokens)

        vectors_array = np.empty(shape=model.wv.syn0.shape)

        mapping = []

        for i, elt in enumerate(model.wv.vocab.items()):
            word = elt[0]
            mapping.append(word)
            vectors_array[i,] = model[word]

        word_vectors = model.wv.syn0
        nb_clusters = 10
        kmeans_clustering = KMeans(n_clusters=nb_clusters)
        idx = kmeans_clustering.fit_predict(word_vectors)

        word_centroid_map = dict(zip(model.wv.index2word, idx))

        centers = kmeans_clustering.cluster_centers_

        for cluster in range(0, nb_clusters):

            print("\nCluster %d" % cluster)

            words = []

            similars = model.similar_by_vector(centers[cluster, :])

            for i in similars:

                words.append(i[0])

        print(words)


    else:
        pass '''

####################################################################
####################################################################
####################################################################

