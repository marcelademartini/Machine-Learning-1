import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Carregamento da base

df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")


# Pré-processamento


# 1. Remover colunas irrelevantes (se tiver, como "id")
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 2. Codificação de variáveis categóricas (transforma string em número)
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# 3. Features (x) e alvo (y) → assumindo que a última coluna seja o alvo
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. Imputação automática (valores ausentes substituídos pela mediana)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)


# Exibir DataFrame

print(df.to_markdown(index=False))
