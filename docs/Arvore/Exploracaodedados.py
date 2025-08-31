# Exploracaodedados.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def carregar_dados():
    df = pd.read_csv(DATA_URL)
    return df

def explorar(df: pd.DataFrame):
    # 1) Natureza dos dados
    print("\n=== Info do dataset ===")
    print(df.info())

    print("\n=== Estatísticas descritivas ===")
    print(df.describe(include="all"))

    print("\n=== Amostra (5 primeiras linhas) ===")
    print(df.head())

    # 2) Distribuição da variável-alvo
    counts = df["Outcome"].value_counts().sort_index()
    print("\n=== Distribuição da classe (Outcome) ===")
    print(counts)

    # Gráfico de barras da classe
    plt.figure(figsize=(6,4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Distribuição da Classe (Outcome)")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_distribuicao_outcome.png", dpi=150)
    plt.close()

    # 3) Histogramas das variáveis numéricas
    num_cols = [c for c in df.columns if df[c].dtype != "O" and c != "Outcome"]
    for col in num_cols:
        plt.figure(figsize=(6,4))
        plt.hist(df[col].dropna(), bins=20)
        plt.title(f"Histograma de {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/hist_{col}.png", dpi=150)
        plt.close()

    # 4) Boxplots (detecção visual de outliers)
    for col in num_cols:
        plt.figure(figsize=(4,4))
        plt.boxplot(df[col].dropna(), vert=True)
        plt.title(f"Boxplot de {col}")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/box_{col}.png", dpi=150)
        plt.close()

    # 5) Correlação (matriz)
    corr = df[num_cols + ["Outcome"]].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, interpolation='nearest')
    plt.title("Matriz de Correlação")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_correlacao.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    df_ = carregar_dados()
    explorar(df_)
    print(f"\nVisualizações salvas em: {OUT_DIR}/")
