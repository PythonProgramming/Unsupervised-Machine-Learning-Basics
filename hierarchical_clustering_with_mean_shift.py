import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

# centers = [[1,1],[5,5]]
# centers = [[1,1],[5,5],[3,8]]
centers = [[1,1],[5,5],[3,10]]

# "_" to ignore labels and note there is a max of 10,000 samples:
# X, _ = make_blobs(n_samples=200, centers=centers, cluster_std=1)
# X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)
# X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=5)
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=0.3)

plt.scatter(X[:,0], X[:,1])
plt.show()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_ # not the same as the "_" labels above
cluster_centers = ms.cluster_centers_
print(cluster_centers)

n_clusters = len(np.unique(labels))

print("Number of estimated clusters:", n_clusters)

colors = 10*['r.','g.','b.','c.','k.','y.','m.']

print(colors)
print(labels)

for i in range(len(X)):
  plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(
  cluster_centers[:,0],cluster_centers[:,1],
  marker="x", s=150, linewidths=5, zorder=10
)

plt.show()
