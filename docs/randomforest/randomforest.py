import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier



# 1) Ler o CSV
df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")  # ajuste o caminho se necessário

# 2) Definir X e y
target_col = "Outcome"
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].values.tolist()
y = df[target_col].tolist()

# 3) Split simples 80/20
indices = list(range(len(X)))
random.seed(42)
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]
X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_test  = [X[i] for i in test_idx]
y_test  = [y[i] for i in test_idx]

# 4) Treinar e avaliar
rf = RandomForest(n_estimators=15, max_depth=8, min_samples_split=4, max_features='sqrt')
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

acc = sum(int(p == t) for p, t in zip(preds, y_test)) / len(y_test)
print("Acurácia:", acc)
