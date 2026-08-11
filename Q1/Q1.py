import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = {
    "Age": [19, 21, 20, 23, 31, 35, 40, 42, 28, 30, 45, 50, 52, 55, 60, 62, 25, 27, 33, 37],
    "Annual Income": [15, 18, 20, 22, 35, 40, 42, 45, 55, 60, 70, 75, 80, 85, 90, 95, 30, 32, 50, 58],
    "Spending Score": [80, 85, 78, 90, 75, 72, 60, 65, 85, 80, 40, 35, 30, 25, 20, 15, 88, 82, 70, 68]
}

df = pd.DataFrame(data)

X = df[["Age", "Annual Income", "Spending Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette_scores = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.plot(range(2, 7), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(range(2, 7), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()

optimal_k = range(2, 7)[silhouette_scores.index(max(silhouette_scores))]

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means and PCA")
plt.show()

print("Optimal Clusters:", optimal_k)
print("Silhouette Score:", max(silhouette_scores))
