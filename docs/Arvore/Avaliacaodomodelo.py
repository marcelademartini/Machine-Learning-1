# Avaliacaodomodelo.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score

from Exploracaodedados import carregar_dados
from Preprocessamento import preprocessar
from Divisaodedados import dividir
from Treinamentodomodelo import treinar_arvore

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_confusion(cm, classes, path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Matriz de Confusão")
    plt.xticks(range(len(classes)), classes, rotation=0)
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

if __name__ == "__main__":
    # Pipeline rápido para garantir consistência com o treino
    df = carregar_dados()
    X, y, imputer, scaler = preprocessar(df)
    X_train, X_test, y_train, y_test = dividir(X, y)

    model = treinar_arvore(X_train, y_train, max_depth=5, min_samples_split=10, min_samples_leaf=5)

    # 5) Avaliação
    y_pred = model.predict(X_test)
    y_proba = getattr(model, "predict_proba", lambda x: None)(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Métricas no conjunto de teste ===")
    print(f"Acurácia : {acc:.3f}")
    print(f"Precisão : {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")

    print("\n=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred, digits=3))

    # Matriz de confusão (plot + save)
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(cm)
    plot_confusion(cm, ["Nao_Diab", "Diab"], f"{OUT_DIR}/05_matriz_confusao.png")

    # Validação cruzada (k=5) – panorama extra
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\nValidação cruzada (k=5) - Acurácia média: {scores.mean():.3f} ± {scores.std():.3f}")

    # Curva ROC (se houver probabilidade)
    if y_proba is not None:
        plt.figure(figsize=(6,5))
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("Curva ROC - Decision Tree")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/06_curva_roc.png", dpi=150)
        plt.close()

    print(f"\nFiguras salvas em: {OUT_DIR}/")
