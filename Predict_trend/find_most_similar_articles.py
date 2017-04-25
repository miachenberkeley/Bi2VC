import pandas as pd
from collections import Counter
import operator
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words



#embedding using tfidf methods
def embedding_tfidf(dictionnary):
    stop_words = get_stop_words('english')
    tfidf = TfidfVectorizer(stop_words = stop_words)
    embedding_array = tfidf.fit_transform(np.concatenate(dictionnary.values))
    embedding_array = embedding_array[:dictionnary.shape[0]]

def cosine_similarities(centroid, mail, n):
    scores = cosine_similarity(centroid, mail)
    if int(round(sorted(scores[:,0], reverse=True)[0])) == 1:
        similar_ids = scores[:,0].argsort()[::-1][1:]
    else:
        similar_ids = scores[:,0].argsort()[::-1]
    return similar_ids[:n], scores