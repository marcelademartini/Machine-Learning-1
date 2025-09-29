import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA

# ===== 1) Carrega o CSV =====
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')

# Define a coluna alvo 
target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]

# X e y (dummies para categóricas)
X_raw = df.drop(columns=[target])
X = pd.get_dummies(X_raw, drop_first=True)
y = df[target]

# ===== Detecta o tipo do problema (classificação vs. regressão) =====
is_numeric_target = np.issubdtype(y.dtype, np.number)
# Se o alvo for categórico, codifica para números (classificação)
if not is_numeric_target:
    y = pd.factorize(y)[0]  # 0..K-1

# Trata NaN
X = X.fillna(X.median(numeric_only=True))

# ===== 2) Split + escala =====
# Para classificação estratificada só faz sentido se houver mais de 1 classe
strat = y if (not is_numeric_target and len(np.unique(y)) > 1) else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ===== Helper: imprimir figura como SVG  =====
def print_svg_current_fig():
    buf = StringIO()
    plt.savefig(buf, format="svg", transparent=True, bbox_inches="tight")
    print(buf.getvalue())
    plt.close()

# ===== 3) Modelo KNN (classificação OU regressão) =====
k = 3
if is_numeric_target:
    # ---------- REGRESSÃO ----------
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)

    # ===== 4R) Métricas de Regressão =====
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("=== MÉTRICAS DE REGRESSÃO ===")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    # ----- Gráfico: y real vs y predito -----
    plt.figure(figsize=(5,4), dpi=120)
    plt.scatter(y_test, y_pred, s=18)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle='--')
    plt.title("Real vs. Predito (Teste)")
    plt.xlabel("Real")
    plt.ylabel("Predito")
    print_svg_current_fig()

    # ----- Gráfico: resíduos -----
    resid = y_test - y_pred
    plt.figure(figsize=(5,4), dpi=120)
    plt.scatter(y_pred, resid, s=18)
    plt.axhline(0, linestyle='--')
    plt.title("Resíduos vs. Predito (Teste)")
    plt.xlabel("Predito")
    plt.ylabel("Resíduo (Real - Predito)")
    print_svg_current_fig()

else:
    # ---------- CLASSIFICAÇÃO ----------
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)

    # ===== 4C) Métricas de Classificação =====
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("=== MÉTRICAS DE CLASSIFICAÇÃO ===")
    print(f"Accuracy            : {acc:.4f}")
    print(f"F1 (macro)          : {f1_macro:.4f}")
    print(f"F1 (weighted)       : {f1_weighted:.4f}")
    print(f"Balanced Accuracy   : {bal_acc:.4f}")
    print(f"Cohen's Kappa       : {kappa:.4f}")
    print(f"Matthews Corrcoef   : {mcc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # ===== Matriz de confusão  =====
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4), dpi=120)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Matriz de Confusão (teste)")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    print_svg_current_fig()

    # ===== ROC-AUC e PR-AUC (se for binário e o modelo suportar probas) =====
    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        try:
            y_proba = knn.predict_proba(X_test_s)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc  = average_precision_score(y_test, y_proba)
            print(f"ROC-AUC (binário): {roc_auc:.4f}")
            print(f"PR-AUC  (binário): {pr_auc:.4f}")

            # Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(5,4), dpi=120)
            plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
            plt.plot([0,1],[0,1],'--')
            plt.title("Curva ROC (Teste)")
            plt.xlabel("FPR (1 - Especificidade)")
            plt.ylabel("TPR (Sensibilidade)")
            plt.legend()
            print_svg_current_fig()

            # Curva Precision-Recall
            pr, rc, _ = precision_recall_curve(y_test, y_proba)
            plt.figure(figsize=(5,4), dpi=120)
            plt.plot(rc, pr, label=f"PR-AUC = {pr_auc:.3f}")
            plt.title("Curva Precision-Recall (Teste)")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            print_svg_current_fig()
        except Exception as e:
            print(f"(Aviso) Não foi possível calcular ROC/PR-AUC: {e}")

# ===== 6) Visualização 2D (PCA) da fronteira de decisão =====
# Mantém a visualização com PCA somente para CLASSIFICAÇÃO,
# pois “fronteira de decisão” não se aplica a regressão.
if X_train.shape[1] >= 2 and not is_numeric_target:
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train_s)
    X_test_2d  = pca.transform(X_test_s)

    knn_viz = KNeighborsClassifier(n_neighbors=k).fit(X_train_2d, y_train)

    h = 0.05
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6,5), dpi=120)
    plt.contourf(xx, yy, Z, alpha=0.30)
    plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train, s=20, marker='o', label='treino')
    plt.scatter(X_test_2d[:,0],  X_test_2d[:,1],  c=y_test,  s=40, marker='x', label='teste')
    plt.title(f"Fronteira de Decisão (PCA 2D) — KNN k={k}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    print_svg_current_fig()
