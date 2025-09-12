import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans

# Usa o CSV como base (duas primeiras colunas numéricas, para manter o mesmo plot)
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')
X_num = df.select_dtypes(include=[np.number]).dropna()

if X_num.shape[1] >= 2:
    X = X_num.iloc[:, :2].to_numpy()
else:
    # Se só houver 1 coluna numérica, duplica para conseguir plotar em 2D
    col = X_num.iloc[:, 0].to_numpy().reshape(-1, 1)
    X = np.hstack([col, col])


plt.figure(figsize=(12, 10))


# Run K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# # Print centroids and inertia
# print("Final centroids:", kmeans.cluster_centers_)
# print("Inertia (WCSS):", kmeans.inertia_)

# # Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())