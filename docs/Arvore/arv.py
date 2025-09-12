import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1) Carregar e explorar dados
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')
# Para usar a base anexada (Training.csv), troque pela linha abaixo:
# df = pd.read_csv('/mnt/data/Training.csv')

# (Exploração) df.describe(), df.dtypes, df.isna().sum() e df.head() ajudam a entender a base.

# 2) Pré-processamento
# Se houver colunas categóricas:
# le = LabelEncoder()
# df['sua_coluna_categ'] = le.fit_transform(df['sua_coluna_categ'].astype(str))
# Tratamento de nulos: df = df.fillna(df.median(numeric_only=True))  # Exemplo

# 3) Divisão em treino/teste
x = df.drop(columns=['Outcome'])   # Features
y = df['Outcome']                  # Alvo binário
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 4) Treinamento do modelo (Decision Tree)
classifier = tree.DecisionTreeClassifier(random_state=42)
classifier.fit(x_train, y_train)

# 5) Avaliação (acurácia + árvore)
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Visualização da árvore
plt.figure(figsize=(12, 10))
tree.plot_tree(classifier)
# Para páginas HTML: exporta em SVG
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
