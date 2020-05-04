import time
import pandas as pd
import multiprocessing as mp
import numpy as np
import psutil
import os
import distance
import sklearn.cluster

num_cores = mp.cpu_count()
print("The kernal has",num_cores, "cores and you can find information regarding mermory usage in",
      psutil.virtual_memory())
start_time = time.time()
items = pd.read_csv("product.csv")
print(os.path.getsize('product.csv'))

items = np.asarray(items)
merged_items = np.concatenate(items, axis=0)

lev_similarity = -1*np.array([[distance.levenshtein(w1, w2)
            for w1 in merged_items] for w2 in merged_items])

affprop = sklearn.cluster.AffinityPropagation(affinity="euclidean", damping=0.5, max_iter=200)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = merged_items[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(merged_items[np.nonzero(affprop.labels_ == cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

print("--- %s seconds ---" % (time.time() - start_time))



