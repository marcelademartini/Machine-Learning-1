import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv")

# Visão geral
print(df.info())
print(df.describe())
print(df["Outcome"].value_counts())

# Histograma
df.hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()

# Mapa de calor de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre variáveis")
plt.show()
