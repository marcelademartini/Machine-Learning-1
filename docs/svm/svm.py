import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

# ===== 1) Ler o CSV =====
df = pd.read_csv('https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv')

# ===== 2) Selecionar duas features numéricas para visualização =====
# (exemplo: duas primeiras colunas, mas você pode trocar pelos nomes)
X = df.iloc[:, :2].values  

# ===== 3) Definir a coluna alvo =====
target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
y_raw = df[target]

# Converter o alvo para -1 e +1
y = np.where(y_raw == y_raw.unique()[0], -1, 1)

# ===== 4) RBF kernel =====
def rbf_kernel(x1, x2, sigma=1):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

# ===== 5) Kernel matrix =====
def kernel_matrix(X, kernel, sigma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j], sigma)
    return K

K = kernel_matrix(X, rbf_kernel, 1)

# ===== 6) Funções de otimização =====
P = np.outer(y, y) * K

def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

def constraint(alpha):
    return np.dot(alpha, y)

cons = {'type': 'eq', 'fun': constraint}
bounds = [(0, None) for _ in range(len(y))]
alpha0 = np.zeros(len(y))

# ===== 7) Resolver o problema dual =====
res = optimize.minimize(objective, alpha0, method='SLSQP', bounds=bounds, constraints=cons)
alpha = res.x

# ===== 8) Vetores de suporte =====
sv_threshold = 1e-5
sv_idx = alpha > sv_threshold

# ===== 9) Calcular bias (b) =====
i = np.where(sv_idx)[0][0]
b = y[i] - np.dot(alpha * y, K[i, :])

# ===== 10) Função de predição =====
def predict(x):
    kx = np.array([rbf_kernel(x, xi, sigma=1) for xi in X])
    return np.dot(alpha * y, kx) + b

# ===== 11) Plot da fronteira =====
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = np.array([predict(np.array([r, c])) for r, c in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#FFDDDD', '#DDDDFF'], alpha=0.8)
plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='--')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class +1')
plt.scatter(X[sv_idx, 0], X[sv_idx, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.title('SVM com Kernel RBF (dados do CSV)')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend()
plt.show()
