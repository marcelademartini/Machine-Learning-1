import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explorar_dados(https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv="Testing.csv"):
    df = pd.read_csv(https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv)

    print("\n Informações do Dataset:")
    print(df.info())
    print("\n Estatísticas Descritivas:")
    print(df.describe())

    # Visualizações
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Glucose"], bins=20, kde=True)
    plt.title("Distribuição da Glicose")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df["BMI"])
    plt.title("Boxplot do IMC")
    plt.show()

    return df
