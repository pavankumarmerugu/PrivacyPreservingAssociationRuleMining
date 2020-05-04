from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
import pandas as pd
import gensim
from gensim.models import Doc2Vec
import multiprocessing as mp
import numpy as np
import psutil
import os
import distance
import sklearn.cluster
from sklearn.metrics import adjusted_rand_score

items = pd.read_csv("product.csv")
items = np.asarray(items)
documents = np.concatenate(items, axis=0)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)


true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Interest group %d:" % i),
    for ind in order_centroids[i, :50]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["hardrock", "movies", "music",])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["cheese", "sports", "football","Yogurt"])
prediction = model.predict(Y)
print(prediction)