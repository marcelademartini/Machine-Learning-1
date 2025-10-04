import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ===== 1) Carrega o CSV =====
df = pd.read_csv("Testing.csv")   # coloque o caminho certo se não estiver na mesma pasta

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

# ===== 4) Matriz de Confusão =====
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", cm)

# ===== 5) Plot Matriz de Confusão =====
plt.figure(figsize=(5,4), dpi=120)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão (teste)")
plt.xlabel("Predito")
plt.ylabel("Real")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
plt.colorbar()
plt.show()

# ===== 6) Relatório de Classificação =====
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, digits=3))
