## 📈 Avaliação do Modelo

!!! example "Resultados da Avaliação"

    Abaixo estão os resultados da avaliação do modelo de árvore de decisão. A acurácia, matriz de confusão e visualização da árvore são fundamentais para entender a performance.

=== "Decision Tree"

    ```python
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    from sklearn import tree

    def avaliar_modelo(modelo, x_test, y_test, feature_names):
        y_pred = modelo.predict(x_test)

        # Acurácia
        acc = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {acc:.2f}")

        # Relatório
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))

        # Matriz de Confusão
        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))

        # Visualizar Árvore
        plt.figure(figsize=(18, 8))
        tree.plot_tree(modelo, feature_names=feature_names, class_names=["Não Diabético", "Diabético"], filled=True)
        plt.show()
    ```

=== "Code - Avaliação"

    ```python
    # Mesmo conteúdo, caso queira repetir ou adaptar
    ```
