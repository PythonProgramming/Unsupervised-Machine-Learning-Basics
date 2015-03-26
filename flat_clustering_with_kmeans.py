import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]
# plt.scatter(x,y)
# plt.show()

X = np.array( zip(x,y) ) # python 2
# X = np.array( list(zip(x,y)) ) # python 3

# kmeans = KMeans(n_clusters=2)
# kmeans = KMeans(n_clusters=3)
kmeans = KMeans(n_clusters=4)
kmf = kmeans.fit(X)
print(type(kmf))
print(kmf)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("centroids:\n%s"%centroids)
print("labels=%s"%labels)

colors = ["g.","r.","c.","y."]

for i in range(len(X)):
  print("coordinate:",X[i], "label:",labels[i])
  plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

plt.show()
