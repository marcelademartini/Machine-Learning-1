# ==========================================
# PROJETO: KNN para Classificação - Dataset IRIS
# ==========================================

# 1) Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2) Carregar dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Mapear números para nomes das espécies
df["species"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# 3) Exploração inicial
print("Primeiras linhas do dataset:")
print(df.head())
print("\nEstatísticas descritivas:")
print(df.describe())
print("\nDistribuição de classes:")
print(df["species"].value_counts())

# Visualização da distribuição por espécie
sns.pairplot(df, hue="species", diag_kind="kde")
plt.show()

# 4) Pré-processamento
X = df.drop("species", axis=1)
y = df["species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5) Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Treinamento do modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 7) Avaliação
y_pred = knn.predict(X_test)

print("\nAcurácia do modelo:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# Visualização da matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Matriz de Confusão - KNN")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# 8) Testar diferentes valores de k
k_values = range(1, 15)
scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker="o")
plt.title("Acurácia vs. Número de Vizinhos (k)")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia")
plt.grid()
plt.show()
