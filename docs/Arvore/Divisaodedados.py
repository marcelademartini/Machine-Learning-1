# -- coding: utf-8 --
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# =========================
# 1) Carregamento da base
# =========================
URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv"
df = pd.read_csv(URL)

# =========================
# 2) Pré-processamento
# =========================
# Remove 'id' se existir
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Codifica diagnosis (B/M) -> 0/1
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])

# Features e alvo
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# =========================
# 3) Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# =========================
# 4) Imputação (mediana)
#    - fit no treino, transform no treino e teste
# =========================
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(strategy="median")

X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols]  = imputer.transform(X_test[num_cols])

# (Opcional) se quiser também preencher o df original só para exibição:
# df[num_cols] = imputer.transform(df[num_cols])

# =========================
# 5) Exibição em Markdown
# =========================
# ATENÇÃO: isto imprime o DataFrame inteiro; se preferir,
# use df.head().to_markdown(index=False) para uma amostra.
print(df.to_markdown(index=False))