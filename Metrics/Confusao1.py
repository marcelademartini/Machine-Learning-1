import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# ===== 1) Carrega CSV =====
df = pd.read_csv("Testing.csv")

# Se existir Outcome, usamos como alvo verdadeiro
y_true = df["Outcome"] if "Outcome" in df.columns else None

# Só pega colunas numéricas
X_num = df.select_dtypes(include=[np.number]).dropna()

if X_num.shape[1] >= 2:
    X = X_num.iloc[:, :2].to_numpy()
else:
    col = X_num.iloc[:, 0].to_numpy().reshape(-1, 1)
    X = np.hstack([col, col])

# ===== 2) KMeans =====
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# ===== 3) Matriz de Confusão =====
if y_true is not None:
    cm = confusion_matrix(y_true, labels)
    print("Matriz de Confusão:\n", cm)

    # Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Clusters (KMeans)")
    plt.ylabel("Classes Reais")
    plt.title("Matriz de Confusão — KMeans vs Outcome")
    plt.show()
else:
    print("⚠️ Nenhuma coluna 'Outcome' encontrada. Não é possível calcular matriz de confusão.")
