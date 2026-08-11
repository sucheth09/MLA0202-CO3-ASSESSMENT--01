import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

X = np.array([
    [13.0, 2.4, 2.0, 19.0, 1.8, 2.8],
    [12.5, 2.2, 2.2, 18.0, 1.9, 2.7],
    [13.2, 2.5, 2.1, 20.0, 1.7, 2.9],
    [12.8, 2.3, 1.9, 19.5, 1.8, 2.8],
    [13.5, 2.6, 2.3, 21.0, 1.6, 3.0],
    [12.7, 2.1, 2.0, 18.5, 2.0, 2.6],
    [11.0, 1.8, 3.0, 15.0, 2.5, 2.0],
    [11.2, 1.9, 3.2, 16.0, 2.4, 2.1],
    [10.8, 1.7, 3.1, 14.5, 2.6, 1.9],
    [11.5, 2.0, 2.9, 16.5, 2.3, 2.2],
    [10.9, 1.8, 3.3, 15.5, 2.7, 2.0],
    [11.3, 1.9, 3.0, 16.2, 2.5, 2.1]
])

y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

print("Original Dataset Shape:", X.shape)

print("\nPCA Output:")
print(X_pca)

print("\nPCA Explained Variance:")
print(pca.explained_variance_ratio_)

print("\nFactor Analysis Output:")
print(X_fa)

print("\nICA Output:")
print(X_ica)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=60)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA")
plt.show()

plt.scatter(X_fa[:, 0], X_fa[:, 1], c=y, cmap="viridis", s=60)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Factor Analysis")
plt.show()

plt.scatter(X_ica[:, 0], X_ica[:, 1], c=y, cmap="viridis", s=60)
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("Independent Component Analysis")
plt.show()
