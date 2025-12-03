import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ===== 1) Carrega o CSV =====
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')

# Define a coluna alvo 
target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]

# X e y (dummies para categóricas)
X_raw = df.drop(columns=[target])
X = pd.get_dummies(X_raw, drop_first=True)
y = df[target]

# Codifica alvo não numérico
if not np.issubdtype(y.dtype, np.number):
    y = pd.factorize(y)[0]

# Trata NaN
X = X.fillna(X.median(numeric_only=True))

# ===== 2) Split + escala =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ===== 3) Treina KNN =====
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

# ===== 4) Métricas =====
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, digits=3))


# ===== Helper: imprimir figura como SVG  =====
def print_svg_current_fig():
    buf = StringIO()
    plt.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
    print(buf.getvalue())
    plt.close()

# ===== 5) Matriz de confusão  =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4), dpi=120)
plt.imshow(cm, interpolation='nearest')
plt.title("Matriz de Confusão (teste)")
plt.xlabel("Predito")
plt.ylabel("Real")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
print_svg_current_fig()

# ===== 6) Visualização 2D (PCA) da fronteira de decisão  =====
if X_train.shape[1] >= 2:
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train_s)
    X_test_2d  = pca.transform(X_test_s)

    knn_viz = KNeighborsClassifier(n_neighbors=k).fit(X_train_2d, y_train)

    h = 0.05
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6,5), dpi=120)
    plt.contourf(xx, yy, Z, alpha=0.30)
    plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train, s=20, marker='o', label='treino')
    plt.scatter(X_test_2d[:,0],  X_test_2d[:,1],  c=y_test,  s=40, marker='x', label='teste')
    plt.title(f"Fronteira de Decisão (PCA 2D) — KNN k={k}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    print_svg_current_fig()
