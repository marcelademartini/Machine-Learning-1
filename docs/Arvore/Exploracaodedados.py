# -- coding: utf-8 --
import matplotlib.pyplot as plt  # mantido, caso você use depois
import pandas as pd
from io import StringIO
from sklearn import tree          # mantido, caso você use depois
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv"

# 1) Carregamento
df = pd.read_csv(URL)

# 2) Pré-visualização básica
print(f"Linhas, colunas: {df.shape}")
print("\nTipos de dados:")
print(df.dtypes)

# 3) Limpeza mínima (não altera a lógica do seu snippet original)
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 4) Info útil sobre a coluna-alvo, se existir
if "diagnosis" in df.columns:
    print("\nContagem de valores em 'diagnosis':")
    print(df["diagnosis"].value_counts(dropna=False))
else:
    print("\nA coluna 'diagnosis' não foi encontrada na base.")

# 5) Exibição em Markdown (apenas amostra para não lotar o terminal)
print("\nAmostra (10 primeiras linhas):")
print(df.head(10).to_markdown(index=False))