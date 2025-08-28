# -- coding: utf-8 --
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# =========================
# 1) Carregamento da base
# =========================
URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv"
df = pd.read_csv(URL)

# =========================
# 2) Pré-processamento
# =========================
# Remoção da coluna 'id', se existir
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Conversão de diagnóstico B/M em 0/1
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])

# Features e alvo
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# =========================
# 3) Imputação de valores ausentes com mediana
# =========================
# Usando SimpleImputer para lidar com qualquer valor nulo nas features numéricas
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy="median")
X[num_cols] = imputer.fit_transform(X[num_cols])

# =========================
# 4) Exibir os dados em Markdown
# =========================
print(df.to_markdown(index=False))