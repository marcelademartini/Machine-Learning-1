from sklearn.model_selection import train_test_split

def dividir_dados(df, test_size=0.2, random_state=42):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
