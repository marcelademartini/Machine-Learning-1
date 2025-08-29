import matplotlib
matplotlib.use("Agg")  # backend não interativo (CI/Pages)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1) Carregar dados
df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")

# 2) Pré-processamento
for col in df.select_dtypes(include=["object"]).columns:
    if col != "Outcome":
        df[col] = LabelEncoder().fit_transform(df[col])

cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, pd.NA)
for col in cols_with_invalid_zeros:
    df[col].fillna(df[col].median(), inplace=True)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 3) Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4) Treinar UMA vez (não sobrescreva o modelo depois)
clf = tree.DecisionTreeClassifier(random_state=42, max_depth=4)
clf.fit(x_train, y_train)

# 5) Acurácia (opcional)
print(f"Accuracy: {clf.score(x_test, y_test):.2f}")

# 6) Plotar e SALVAR como arquivo estático
fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Não", "Sim"],
    filled=True,
    rounded=True,
    ax=ax
)
fig.tight_layout()

# >>> Ajuste o caminho abaixo para a pasta que o seu Pages publica <<<
out_svg = "docs/arvore_decisao.svg"
out_png = "docs/arvore_decisao.png"
fig.savefig(out_svg, format="svg", bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

# 7) (Opcional) Gerar um HTML com o SVG embutido
from io import StringIO
buf = StringIO()
# Reusa a figura: salve novamente no buffer
fig2, ax2 = plt.subplots(figsize=(20, 10))
tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Não", "Sim"],
    filled=True,
    rounded=True,
    ax=ax2
)
fig2.tight_layout()
fig2.savefig(buf, format="svg", bbox_inches="tight")
plt.close(fig2)

svg_txt = buf.getvalue()
with open("docs/arvore.html", "w", encoding="utf-8") as f:
    f.write(f"<!doctype html><meta charset='utf-8'>\n{svg_txt}")

print("Arquivos gerados: docs/arvore_decisao.svg, docs/arvore_decisao.png, docs/arvore.html")
