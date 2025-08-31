from sklearn.tree import DecisionTreeClassifier

def treinar_modelo(x_train, y_train):
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(x_train, y_train)
    return modelo

# Exemplo:
# modelo = treinar_modelo(x_train, y_train)
