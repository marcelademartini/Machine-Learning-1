import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# ===== Helper: imprimir figura como SVG (mesmo do seu código) =====
def print_svg_current_fig():
    buf = StringIO()
    plt.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
    print(buf.getvalue())
    plt.close()

# ===== 1) Carrega o CSV =====
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')

# Garante alvo verdadeiro (para comparar com clusters)
if 'Outcome' not in df.columns:
    raise ValueError("Não há coluna 'Outcome' no CSV — sem rótulos reais não dá para fazer matriz de confusão.")

y_true = df['Outcome'].to_numpy()

# ===== 2) Seleciona features numéricas (como no seu K-Means) =====
X_num = df.select_dtypes(include=[np.number]).dropna()

if X_num.shape[1] >= 2:
    X = X_num.iloc[:, :2].to_numpy()
else:
    col = X_num.iloc[:, 0].to_numpy().reshape(-1, 1)
    X = np.hstack([col, col])

# ===== 3) Roda K-Means =====
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# ===== 4) MATRIZ DE CONFUSÃO (BRUTA): classes reais x clusters =====
cm_raw = confusion_matrix(y_true, labels)
print("Matriz de Confusão (bruta) — y_true x cluster:\n", cm_raw)

plt.figure(figsize=(5,4), dpi=120)
plt.imshow(cm_raw, interpolation='nearest')
plt.title("Matriz de Confusão (bruta)\n(Classes reais × Clusters)")
plt.xlabel("Cluster (predito pelo K-Means)")
plt.ylabel("Classe real (Outcome)")
for i in range(cm_raw.shape[0]):
    for j in range(cm_raw.shape[1]):
        plt.text(j, i, str(cm_raw[i, j]), ha="center", va="center")
plt.colorbar()
print_svg_current_fig()

# ===== 5) (OPCIONAL) REMAPEAMENTO DOS CLUSTERS PARA CLASSES =====
# Tenta encontrar a melhor permutação de mapeamento cluster->classe que maximize acertos
classes = np.unique(y_true)
n_classes = len(classes)
n_clusters = k

# Se número de clusters >= número de classes, tentamos mapear as 'n_classes' primeiras posições
best_perm = None
best_hits = -1

for perm in permutations(range(n_clusters), n_classes):
    # constrói predição remapeando apenas os clusters usados (demais ficam como a primeira classe)
    map_dict = {cluster_idx: classes[i] for i, cluster_idx in enumerate(perm)}
    y_mapped = np.array([map_dict.get(c, classes[0]) for c in labels])
    hits = (y_mapped == y_true).sum()
    if hits > best_hits:
        best_hits = hits
        best_perm = perm

# Aplica melhor mapeamento encontrado
map_dict = {cluster_idx: classes[i] for i, cluster_idx in enumerate(best_perm)}
y_pred_mapped = np.array([map_dict.get(c, classes[0]) for c in labels])

cm_mapped = confusion_matrix(y_true, y_pred_mapped)
print("\nMatriz de Confusão (clusters remapeados → classes):\n", cm_mapped)
print("Melhor permutação cluster->classe encontrada:", map_dict)

plt.figure(figsize=(5,4), dpi=120)
plt.imshow(cm_mapped, interpolation='nearest')
plt.title("Matriz de Confusão (clusters remapeados)")
plt.xlabel("Predito (após remapeamento)")
plt.ylabel("Classe real (Outcome)")
for i in range(cm_mapped.shape[0]):
    for j in range(cm_mapped.shape[1]):
        plt.text(j, i, str(cm_mapped[i, j]), ha="center", va="center")
plt.colorbar()
print_svg_current_fig()

# ===== 6) (Opcional) Plot do clustering em 2D, como no seu exemplo =====
plt.figure(figsize=(6,5), dpi=120)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, alpha=0.8, label='pontos', cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='centroides')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
print_svg_current_fig()
