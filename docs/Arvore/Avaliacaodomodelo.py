import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabulate import tabulate  


# 1. Carregamento da base

df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")


# 2. Pré-processamento


# Converte colunas categóricas em números se houver
for col in df.select_dtypes(include=["object"]).columns:
    if col != "Outcome":  # evitar codificar a variável alvo
        df[col] = LabelEncoder().fit_transform(df[col])

# Substitui zeros inválidos por NaN em colunas que não podem ter 0
cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, pd.NA)

# Preenche valores ausentes com a mediana
for col in cols_with_invalid_zeros:
    df[col].fillna(df[col].median(), inplace=True)

# Features (X) e Target (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]


# 3. Divisão treino e teste

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# 4. Modelo de Árvore

classifier = tree.DecisionTreeClassifier(random_state=42, max_depth=4)
classifier.fit(x_train, y_train)


# 5. Avaliação

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())