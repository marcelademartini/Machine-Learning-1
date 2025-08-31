# Treinamentodomodelo.py
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from Exploracaodedados import carregar_dados, explorar
from Preprocessamento import preprocessar
from Divisaodedados import dividir

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def treinar_arvore(X_train, y_train, **kwargs):
    # Hiperparâmetros default + possibilidade de ajuste
    model = DecisionTreeClassifier(
        random_state=42,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # 1) Exploração (gera visualizações)
    df = carregar_dados()
    explorar(df)

    # 2) Pré-processamento
    X, y, imputer, scaler = preprocessar(df)

    # 3) Divisão
    X_train, X_test, y_train, y_test = dividir(X, y)

    # 4) Treinamento do modelo (ajuste leve para reduzir overfitting)
    model = treinar_arvore(X_train, y_train, max_depth=5, min_samples_split=10, min_samples_leaf=5)

    # Salva uma imagem da árvore treinada
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Nao_Diab", "Diab"],
        filled=True
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_arvore_treinada.png", dpi=200)
    plt.close()

    # Salva importâncias de features
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Importância das variáveis ===")
    print(importances)

    plt.figure(figsize=(8,5))
    plt.bar(importances.index, importances.values)
    plt.title("Importância das variáveis (Decision Tree)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_importancias.png", dpi=150)
    plt.close()

    print(f"\nArquivos salvos em {OUT_DIR}/. Agora rode Avaliacaodomodelo.py para ver as métricas.")
