import matplotlib
matplotlib.use("Agg")  # usa backend "sem tela" (necessário no GitHub Pages ou servidor)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1) Carregar base
df = pd.read_csv("https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv")

# 2) Pré-processamento
df = df.drop(columns=["id"])  # coluna irrelevante
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])

# Imputação de valores ausentes
df["concavity_mean"].fillna(df["concavity_mean"].median(), inplace=True)
df["concave points_mean"].fillna(df["concave points_mean"].median(), inplace=True)

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# 3) Divisão treino/teste
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4) Treinar árvore
clf = tree.DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(x_train, y_train)

# 5) Plotar árvore
fig, ax = plt.subplots(figsize=(22, 12))
tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Benigno", "Maligno"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)

# 6) Salvar imagem (PNG e SVG)
fig.savefig("arvore.png", dpi=300, bbox_inches="tight")
fig.savefig("arvore.svg", format="svg", bbox_inches="tight")
plt.close(fig)

print("Árvore de decisão salva como 'arvore.png' e 'arvore.svg'")
