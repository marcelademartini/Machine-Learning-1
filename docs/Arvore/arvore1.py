import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dataset
df = pd.read_csv("Training.csv")

# Mostrar informações gerais
print(df.info())
print(df.describe())

# Distribuição da variável alvo
plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df, palette="viridis")
plt.title("Distribuição do Outcome (Diabetes)")
plt.xlabel("Diabetes (0 = Não, 1 = Sim)")
plt.ylabel("Contagem")
plt.show()

# Histograma de variáveis numéricas
df.hist(bins=20, figsize=(14,10), color="teal")
plt.suptitle("Distribuição das Variáveis", fontsize=16)
plt.show()

# Matriz de correlação
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()
