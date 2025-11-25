import os
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt


# ===== CSV loader =====
RAW_URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/main/Testing.csv"
LOCAL_CANDIDATES = [
    "Testing.csv",
    os.path.join("docs", "svm", "Testing.csv"),
    os.path.join("docs", "Testing.csv"),
]

def fetch_csv_with_retries(url, local_candidates=None, retries=3, timeout=10):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        if local_candidates:
            for p in local_candidates:
                if os.path.exists(p):
                    return pd.read_csv(p)
        raise RuntimeError(f"Failed to load CSV from '{url}' and no local fallback found. Place 'Testing.csv' in the repo.")

df = fetch_csv_with_retries(RAW_URL, local_candidates=LOCAL_CANDIDATES)

# ===========================
# 1. Selecionar variáveis (features) e alvo (target)
# ===========================
# manter apenas as colunas numéricas e exigir pelo menos duas colunas para a demonstração em 2D
numeric = df.select_dtypes(include=[np.number])
if numeric.shape[1] < 2:
    raise RuntimeError("Need at least two numeric features for this SVM demo. Put Testing.csv with numeric columns in the repo.")

X = numeric.iloc[:, :2].values  # primeiras duas variáveis numéricas
y_raw = df.iloc[:, -1].values  # última coluna do dataframe original usada como rótulo

# Converter rótulos para binário {-1, +1}
unique_values = np.unique(y_raw)
if len(unique_values) != 2:
    raise ValueError("The SVM demo requires exactly two classes in the target column.")
y = np.where(y_raw == unique_values[0], -1, 1).astype(float)

# ===========================
# 2. Kernel RBF
# ===========================
def rbf_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2.0 * sigma**2))

def kernel_matrix(X, kernel, sigma=1.0):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], sigma)
    return K

sigma = 1.0
K = kernel_matrix(X, rbf_kernel, sigma)

# ===========================
# 3. Otimização dual (com restrições de caixa 0 <= alpha <= C)
# ===========================
C = 1.0  # parâmetro soft-margin; altere se necessário
P = np.outer(y, y) * K

def objective(alpha):
    return 0.5 * np.dot(alpha, P.dot(alpha)) - np.sum(alpha)

def constraint(alpha):
    return np.dot(alpha, y)

cons = {'type': 'eq', 'fun': constraint}
bounds = [(0.0, C) for _ in range(len(y))]
alpha0 = np.zeros(len(y))

res = optimize.minimize(objective, alpha0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
if not res.success:
    print("Warning: optimization did not converge:", res.message)
alpha = np.clip(res.x, 0.0, C)

# ===========================
# 4. Support vectors e bias
# ===========================
sv_threshold = 1e-6
sv_idx = alpha > sv_threshold
sv_indices = np.where(sv_idx)[0]
if sv_indices.size == 0:
    raise RuntimeError("No support vectors found. Try increasing C or check data.")

# calcular o viés b como a média sobre os vetores de suporte
b_vals = []
for i in sv_indices:
    b_i = y[i] - np.sum(alpha * y * K[:, i])
    b_vals.append(b_i)
b = float(np.mean(b_vals))

# ===========================
# 5. Função de decisão / predição
# ===========================
def decision_function(x):
    kx = np.array([rbf_kernel(x, xi, sigma=sigma) for xi in X])
    return np.dot(alpha * y, kx) + b

def predict_label(x):
    return np.sign(decision_function(x))

# ===========================
# 6. Plot decision boundary + support vectors
# ===========================
step = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([decision_function(pt) for pt in grid_points])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#FFDDDD', '#DDDDFF'], alpha=0.8)
plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='--')

plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label=f'Classe {unique_values[0]} (-1)')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label=f'Classe {unique_values[1]} (+1)')

plt.scatter(X[sv_idx, 0], X[sv_idx, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.xlabel(numeric.columns[0])
plt.ylabel(numeric.columns[1])
plt.title('SVM com Kernel RBF (dataset Testing.csv)')
plt.legend()
plt.tight_layout()
plt.show()

