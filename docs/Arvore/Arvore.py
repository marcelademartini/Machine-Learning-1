import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.figure(figsize=(12, 10))

# 1) Carregar o conjunto de dados
df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")

# 2) PRÉ-PROCESSAMENTO
#    - Zeros clinicamente inválidos tratados como ausentes (NaN)
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

#    - Separar variáveis independentes (X) e alvo (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#    - LabelEncoder no alvo apenas se for string (não é o caso aqui, mas deixamos a salvaguarda)
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# 3) DIVISÃO DOS DADOS (treino e teste) com estratificação para manter proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) TREINAMENTO DO MODELO (Decision Tree) com Pipeline:
#    - Imputer (mediana) -> Scaler (normalização) -> DecisionTreeClassifier
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", tree.DecisionTreeClassifier(max_depth=5, random_state=42))
])

pipeline.fit(X_train, y_train)

# 5) AVALIAÇÃO DO MODELO
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}\n")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# 6) VISUALIZAÇÃO DA ÁRVORE
#    Observação: os thresholds estarão em escala padronizada (z-score) devido ao StandardScaler.
clf = pipeline.named_steps["clf"]
tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Não Diabético", "Diabético"],
    filled=True
)

# Exportar a figura como SVG (texto) para possível uso em HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())