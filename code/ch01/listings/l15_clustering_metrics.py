from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score)

X, y = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=7)
for k in range(2, 7):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(f"k={k}  sil={silhouette_score(X, lab):.3f}  "
          f"CH={calinski_harabasz_score(X, lab):6.0f}  "
          f"DB={davies_bouldin_score(X, lab):.3f}  "
          f"ARI={adjusted_rand_score(y, lab):.3f}  "
          f"NMI={normalized_mutual_info_score(y, lab):.3f}")
