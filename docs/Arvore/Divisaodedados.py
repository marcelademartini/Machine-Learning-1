import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabulate import tabulate  

# ==============================
# Carregamento da base
# ==============================
df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")

# ==============================
# Pré-processamento
# ==============================

# Converte todas as colunas categóricas em numéricas
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Define as features (X) e o alvo (y)
# A base tem a coluna "prognosis" como alvo
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Imputação de valores ausentes com a mediana
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# ==============================
# Divisão em treino e teste
# ==============================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================
# Exibir DataFrame formatado
# ==============================
print(df.to_markdown(index=False))
