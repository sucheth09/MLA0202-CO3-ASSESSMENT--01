import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X = np.array([
    [1.0, 1.2], [1.2, 1.0], [0.8, 1.1], [1.1, 0.9],
    [1.3, 1.4], [0.9, 1.3], [1.2, 1.3], [1.0, 0.8],
    [5.0, 5.2], [5.2, 5.0], [4.8, 5.1], [5.1, 4.9],
    [5.3, 5.4], [4.9, 5.3], [5.2, 5.3], [5.0, 4.8],
    [9.0, 1.0], [9.2, 1.2], [8.8, 0.9], [9.1, 1.1],
    [9.3, 1.3], [8.9, 1.2], [9.2, 0.8], [9.0, 1.3]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

kmeans_score = silhouette_score(X_scaled, kmeans_labels)
gmm_score = silhouette_score(X_scaled, gmm_labels)

print("K-Means Silhouette Score:", kmeans_score)
print("GMM Silhouette Score:", gmm_score)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering")
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="viridis", s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Gaussian Mixture Model")
plt.show()
