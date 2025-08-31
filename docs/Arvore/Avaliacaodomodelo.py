from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

def avaliar_modelo(modelo, x_test, y_test, feature_names):
    y_pred = modelo.predict(x_test)
    
    # Acurácia
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.2f}")
    
    # Relatório e matriz
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # Visualização da árvore
    plt.figure(figsize=(18, 8))
    tree.plot_tree(modelo, feature_names=feature_names, class_names=["Não Diabético", "Diabético"], filled=True)
    plt.savefig("docs/assets/arvore.png")  
