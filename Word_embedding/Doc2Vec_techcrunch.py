import gensim
import numpy as np
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plotWords(useModel):
    # get model, we use w2v only
    w2v, d2v = useModel()

    words_np = []
    # a list of labels (words)
    words_label = []
    for word in w2v.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)
    print('Added %s words. Shape %s' % (len(words_np), np.shape(words_np)))

    pca = PCA(n_components=2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)

    # plt.plot(pca.explained_variance_ratio_)
    for index, vec in enumerate(reduced):
        # print ('%s %s'%(words_label[index],vec))
        if index < 100:
            x, y = vec[0], vec[1]
            plt.scatter(x, y)
            plt.annotate(words_label[index], xy=(x, y))
    plt.show()

documents = np.load("/Users/chen/Desktop/Bi2VC/data/tokens/l_tokens.npy")
documents = documents[()]



taggeddoc = []


for index, article in enumerate(documents):
    td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(article))).split(), str(index))
    #print(td)
    taggeddoc.append(td)
print ('Data Loading finished')

print (len(documents), type(documents))
#print(taggeddoc)


# build the model
model = gensim.models.Doc2Vec(taggeddoc, dm=0, alpha=0.025, size=20, min_alpha=0.025, min_count=0)
print(model)
print(len(taggeddoc))

#model.train(taggeddoc)

# start training
for epoch in range(10):
    if epoch % 100 == 0:
        print ('Now training epoch %s' % epoch)
    model.train(taggeddoc)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


#model.train(taggeddoc)
print(model)
# shows the similar words
print (model.most_similar('ceo'))

# shows the learnt embedding
#print (model['suppli'])

# shows the similar docs with id = 2
#print (model.docvecs.most_similar(str(2)))


model.save('trained.model')
model.wv.save_word2vec_format('trained.word2vec')

print('Loading word vectors')
word_vectors = model.wv.syn0
#num_clusters = word_vectors.shape[0] / 5

print('Fitting k-means')
# Initalize a k-means object and use it to extract centroids
nb_clusters = 100
kmeans_clustering = KMeans( n_clusters = nb_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

print('K-means fitted')

word_centroid_map = dict(zip( model.wv.index2word, idx ))
#print(word_centroid_map)
centers = kmeans_clustering.cluster_centers_
#word_list = list(word_centroid_map.values())
#word_keys = list(word_centroid_map.keys())

for cluster in range(0, nb_clusters):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    # for word, clust in word_centroid_map.items():
    #     if(clust==cluster and len(words) < 5):
    #         words.append(word)
    #print(words)
    similars = model.similar_by_vector(centers[cluster,:])
    for i in similars:
        words.append(i[0])
    print(words)


#print(kmeans_clustering.cluster_centers_)

#print(model.similar_by_vector(centers[0,:]))
#for i in range(0, nb_clusters):
    #print(centers[i,:])



#for i in range(0, len(centers)):
#    print(model.most_similar(centers[i]));

#print(centers)

#print(kmeans_clustering.labels_)
