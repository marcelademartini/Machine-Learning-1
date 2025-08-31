from sklearn.model_selection import train_test_split

def dividir_dados(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

# Exemplo de uso
# x_train, x_test, y_train, y_test = dividir_dados(x, y)
