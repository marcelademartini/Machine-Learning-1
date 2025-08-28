import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# 1. Exploração dos Dados
# =============================
df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")

print("\n=== Informações Gerais ===")
print(df.info())
print("\n=== Primeiras Linhas ===")
print(df.head())
print("\n=== Estatísticas Descritivas ===")
print(df.describe())

# Visualizações
plt.figure(figsize=(10, 6))
sns.histplot(df["Glucose"], kde=True)
plt.title("Distribuição da Glicose")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

# =============================
# 2. Pré-processamento
# =============================
cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, pd.NA)

# Imputação com a mediana
imputer = SimpleImputer(strategy="median")
df[cols_with_invalid_zeros] = imputer.fit_transform(df[cols_with_invalid_zeros])

# Normalização
scaler = StandardScaler()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_scaled = scaler.fit_transform(X)

# =============================
# 3. Divisão dos Dados
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 4. Treinamento do Modelo
# =============================
model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# Visualizar a árvore
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.show()

# =============================
# 5. Avaliação do Modelo
# =============================
y_pred = model.predict(X_test)

print("\n=== Acurácia ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred))

print("\n=== Matriz de Confusão ===")
print(confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.show()

# =============================
# 6. Relatório Final
# =============================
print("\n=== Relatório Final ===")
print("O modelo Decision Tree foi treinado com profundidade máxima = 4.")
print("Resultados mostram acurácia próxima de", round(accuracy_score(y_test, y_pred), 2))
print("Possíveis melhorias incluem:")
print("- Testar outros algoritmos (Random Forest, XGBoost).")
print("- Ajustar hiperparâmetros (profundidade, critério, min_samples_split).")
print("- Realizar cross-validation para validar melhor a performance.")
