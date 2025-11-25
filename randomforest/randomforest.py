import os
import io
import math
import random
import requests
import numpy as np
import pandas as pd
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ===== 1) Ler o CSV (robusto) =====
URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/main/Testing.csv"
LOCAL_CANDIDATES = [
    "Testing.csv",
    os.path.join("docs", "randomforest", "Testing.csv"),
    os.path.join("docs", "Testing.csv"),
]

random.seed(42)
np.random.seed(42)


def fetch_csv_with_retries(url, local_candidates=None, retries=3, timeout=10):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        # try local candidates
        if local_candidates:
            for p in local_candidates:
                if os.path.exists(p):
                    return pd.read_csv(p)
        raise RuntimeError(
            f"Failed to load CSV from URL '{url}' ({e}) and no local fallback found. "
            "Place 'Testing.csv' in the repository (repo root or docs/randomforest/) or try again later."
        )


df = fetch_csv_with_retries(URL, local_candidates=LOCAL_CANDIDATES)


# ===== 2) Implementação do RandomForest =====
def gini_impurity(y):
    if not y:
        return 0.0
    counts = Counter(y)
    impurity = 1.0
    n = len(y)
    for count in counts.values():
        p = count / n
        impurity -= p * p
    return impurity


def split_dataset(X, y, feature_idx, value):
    left_X, left_y, right_X, right_y = [], [], [], []
    for xi, yi in zip(X, y):
        if xi[feature_idx] <= value:
            left_X.append(xi)
            left_y.append(yi)
        else:
            right_X.append(xi)
            right_y.append(yi)
    return left_X, left_y, right_X, right_y


class Node:
    def __init__(self, feature_idx=None, value=None, left=None, right=None, label=None):
        self.feature_idx = feature_idx
        self.value = value
        self.left = left
        self.right = right
        self.label = label


def build_tree(X, y, max_depth, min_samples_split, max_features):
    # leaf
    if len(y) < min_samples_split or max_depth == 0:
        return Node(label=Counter(y).most_common(1)[0][0])

    n_features = len(X[0])
    features = random.sample(range(n_features), min(max_features, n_features))

    best_gini = float("inf")
    best = None

    for fi in features:
        values = sorted(set(row[fi] for row in X))
        for v in values:
            left_X, left_y, right_X, right_y = split_dataset(X, y, fi, v)
            if not left_y or not right_y:
                continue
            p_left = len(left_y) / len(y)
            g = p_left * gini_impurity(left_y) + (1 - p_left) * gini_impurity(right_y)
            if g < best_gini:
                best_gini = g
                best = (fi, v, left_X, left_y, right_X, right_y)

    if best is None:
        return Node(label=Counter(y).most_common(1)[0][0])

    fi, v, lX, ly, rX, ry = best
    left = build_tree(lX, ly, max_depth - 1, min_samples_split, max_features)
    right = build_tree(rX, ry, max_depth - 1, min_samples_split, max_features)
    return Node(feature_idx=fi, value=v, left=left, right=right)


def predict_tree(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature_idx] <= node.value:
        return predict_tree(node.left, x)
    return predict_tree(node.right, x)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)
        n_features = len(X[0])
        if self.max_features == 'sqrt':
            mf = max(1, int(math.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            mf = min(self.max_features, n_features)
        else:
            mf = n_features

        for _ in range(self.n_estimators):
            idxs = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            Xb = [X[i] for i in idxs]
            yb = [y[i] for i in idxs]
            tree = build_tree(Xb, yb, self.max_depth, self.min_samples_split, mf)
            self.trees.append(tree)

    def predict(self, X):
        preds = []
        for x in X:
            votes = [predict_tree(t, x) for t in self.trees]
            preds.append(Counter(votes).most_common(1)[0][0])
        return preds


# ===== 3) Preparar dados, treinar e avaliar =====
target_col = "Outcome"
if target_col not in df.columns:
    raise RuntimeError(f"Target column '{target_col}' not found in dataframe.")

feature_cols = [c for c in df.columns if c != target_col]

# converter para numérico quando possível e preencher NaNs
X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
X_df = X_df.fillna(X_df.median(numeric_only=True))
y_series = df[target_col]
# if y not numeric, encode to integers
if not pd.api.types.is_integer_dtype(y_series) and not pd.api.types.is_bool_dtype(y_series):
    y_series = pd.Categorical(y_series).codes

X = X_df.values.tolist()
y = y_series.tolist()

n = len(X)
random.seed(42)
idx = list(range(n))
random.shuffle(idx)
split_idx = math.floor(n * 0.8)
train_idx, val_idx = idx[:split_idx], idx[split_idx:]

X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_val = [X[i] for i in val_idx]
y_val = [y[i] for i in val_idx]

# train custom RF
rf = RandomForest(n_estimators=15, max_depth=6, min_samples_split=4, max_features='sqrt')
rf.fit(X_train, y_train)
preds = rf.predict(X_val)

acc = sum(int(p == t) for p, t in zip(preds, y_val)) / len(y_val)
print(f"Acurácia de validação: {acc:.3f}")

# evaluate
X_test = X_val
y_test = y_val

y_pred = rf.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
print("\nRelatório de Classificação (HTML):")
print(report_df.to_html(classes="table table-sm", border=0, index=True))

labels = sorted(list(set(y_test)))
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nMatriz de Confusão (HTML):")
print(cm_df.to_html(classes="table table-sm", border=0))

# ===== Feature importances usando sklearn =====
try:
    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train)
    skrf = RandomForestClassifier(n_estimators=100, random_state=42)
    skrf.fit(X_train_arr, y_train_arr)
    imp = pd.Series(skrf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importances (sklearn RandomForestClassifier):")
    print(imp.to_frame(name="importance").to_html(border=0))
except Exception as e:
    print("\nCould not compute sklearn feature importances:", e)

   
