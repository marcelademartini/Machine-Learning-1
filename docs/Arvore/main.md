# 🌳 Diagnóstico de Diabetes com Árvores de Decisão

---

## 📊 Exploração dos Dados

!!! example "Sobre a base de dados"
    O dataset contém dados clínicos de pacientes, como número de gestações, glicose, pressão arterial, insulina, idade, entre outros. O objetivo é prever se um paciente tem diabetes com base nesses atributos.

=== "Código"

    ```python
    --8<-- "docs/Arvore/Exploracaodedados.py"
    ```

---

## 🧼 Pré-processamento

!!! example "Tratamento e limpeza dos dados"
    Colunas com valores zero (Glucose, BloodPressure, SkinThickness, Insulin, BMI) foram tratadas com a mediana. O modelo foi normalizado para que os algoritmos funcionem com dados na mesma escala.

=== "Código"

    ```python
    --8<-- "docs/Arvore/Preprocessamento.py"
    ```

---

## ✂️ Divisão dos Dados

!!! example "Separação entre treino e teste"
    Os dados foram divididos em 80% treino e 20% teste, garantindo uma amostra aleatória mas reprodutível com `random_state=42`.

=== "Código"

    ```python
    --8<-- "docs/Arvore/Divisaodedados.py"
    ```

---

## 🤖 Treinamento do Modelo

!!! example "Criação da Árvore de Decisão"
    Um modelo de árvore de decisão foi treinado com os dados pré-processados usando a biblioteca `scikit-learn`.

=== "Código"

    ```python
    --8<-- "docs/Arvore/Treinamentodomodelo.py"
    ```

---

## 📈 Avaliação do Modelo

!!! example "Resultados da Avaliação"
    Foram avaliados a acurácia, a matriz de confusão, o relatório de classificação e a visualização da árvore.

### 📊 Métricas

=== "Código da Avaliação"

    ```python
    --8<-- "docs/Arvore/Avaliacaodomodelo.py"
    ```

### 🌳 Árvore de Decisão Gerada

> O gráfico abaixo foi salvo via `plt.savefig("docs/assets/arvore.png")` e renderizado aqui como imagem.

![Árvore de Decisão](../assets/arvore.png)

---

## ✅ Conclusão

Este projeto demonstrou como usar Árvores de Decisão para prever a presença de diabetes. O modelo apresentou resultados satisfatórios, e a árvore gerada facilita a interpretação. Melhorias futuras podem incluir ajuste de hiperparâmetros ou uso de Random Forest.
