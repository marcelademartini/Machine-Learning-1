import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1. Carregar o dataset
# -------------------------------
url = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv"
df = pd.read_csv(url)

print("Formato do dataset:", df.shape)
print(df.head())

# -------------------------------
# 2. Separar features (X) e, se quiser, o alvo (y)
# -------------------------------
feature_cols = [col for col in df.columns if col != "Outcome"]
X = df[feature_cols].values

# y não é usado no PageRank, mas deixo separado se você quiser analisar depois
if "Outcome" in df.columns:
    y = df["Outcome"].values
else:
    y = None

# -------------------------------
# 3. Padronizar os dados
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Construir o grafo via k-vizinhos mais próximos
#    Cada linha do dataset é um nó.
# -------------------------------
n_samples = X_scaled.shape[0]
k = 5  # número de vizinhos para criar arestas 

# NearestNeighbors para achar vizinhos em espaço de atributos
nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")  # +1 porque inclui o próprio ponto
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)

# Matriz de adjacência A (n_samples x n_samples)
# A[i, j] = 1 se existe uma aresta de i -> j
A = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    # indices[i, 0] é o próprio ponto i, então pulamos
    neighbors = indices[i, 1:]
    for j in neighbors:
        A[i, j] = 1.0

# -------------------------------
# 5. Construir a matriz de transição P para o PageRank
# -------------------------------
# Queremos P como matriz de probabilidades por linha (row-stochastic)
P = A.copy()
row_sums = P.sum(axis=1)

# Tratar nós "pendurados" (sem saída): distribui uniformemente entre todos
for i in range(n_samples):
    if row_sums[i] == 0:
        P[i, :] = 1.0 / n_samples
    else:
        P[i, :] = P[i, :] / row_sums[i]

# -------------------------------
# 6. Implementar o PageRank via iteração de potência
# -------------------------------
alpha = 0.85  # fator de amortecimento (damping factor)
tol = 1e-6    # tolerância para convergência
max_iter = 100

# Vetor inicial de PageRank (distribuição uniforme)
pagerank = np.ones(n_samples) / n_samples

for it in range(max_iter):
    # r_{t+1} = alpha * P^T * r_t + (1 - alpha) * v
    # onde v é distribuição de teleporte uniforme (1/N)
    new_pagerank = alpha * P.T.dot(pagerank) + (1 - alpha) * (1.0 / n_samples)

    # Verificar convergência
    diff = np.linalg.norm(new_pagerank - pagerank, 1)
    pagerank = new_pagerank

    # print(f"Iteração {it+1}, diferença L1 = {diff}")
    if diff < tol:
        # print("Convergiu na iteração:", it+1)
        break

# Normalizar (só para garantir que some 1)
pagerank = pagerank / pagerank.sum()

# -------------------------------
# 7. Juntar o resultado ao DataFrame
# -------------------------------
df["PageRank"] = pagerank

print("\nDataFrame com coluna de PageRank:")
print(df.head())

# -------------------------------
# 8. Exemplo: mostrar os 10 pacientes mais "importantes" segundo PageRank
# -------------------------------
df_sorted = df.sort_values("PageRank", ascending=False)
print("\nTop 10 linhas com maior PageRank:")
print(df_sorted.head(10))

# ============================================================
# 9. HISTOGRAMA DA DISTRIBUIÇÃO DO PAGERANK
# ============================================================
plt.figure(figsize=(8, 4))
plt.hist(df["PageRank"], bins=20)
plt.title("Distribuição dos Scores de PageRank")
plt.xlabel("PageRank")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# ============================================================
# 10. TOP 20 PAGERANK – GRÁFICO DE BARRAS
# ============================================================
top20 = df_sorted.head(20)

plt.figure(figsize=(10, 5))
plt.bar(range(20), top20["PageRank"])
plt.title("Top 20 Maiores Valores de PageRank")
plt.xlabel("Índice da Amostra")
plt.ylabel("PageRank")
plt.xticks(range(20), top20.index, rotation=90)
plt.tight_layout()
plt.show()

# ============================================================
# 11. PCA 2D COLORIDO PELO PAGERANK
# ============================================================
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["PageRank"])
plt.colorbar(scatter, label="PageRank")
plt.title("Visualização PCA Colorida por PageRank")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
