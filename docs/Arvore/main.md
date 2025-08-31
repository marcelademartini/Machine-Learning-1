# Diagnóstico de Diabetes com Árvores de Decisão

---

## 📊 Exploração dos Dados

!!! example "Descrição do dataset"
    O diabetes é uma das doenças crônicas mais comuns no mundo. O dataset utilizado contém dados clínicos de pacientes com o objetivo de prever a variável `Outcome` (0 = não diabético, 1 = diabético).

=== "Código de exploração"

    ```python
    --8<-- "docs/Arvore/Exploracaodedados.py"
    ```

---

## 🧼 Pré-processamento

!!! example "Explicação do tratamento dos dados"
    Foram substituídos valores 0 por mediana em colunas clínicas e aplicada normalização para melhorar o desempenho dos algoritmos.

=== "Código de pré-processamento"

    ```python
    --8<-- "docs/Arvore/Preprocessamento.py"
    ```

---

## ✂️ Divisão dos Dados

=== "Código de divisão"

    ```python
    --8<-- "docs/Arvore/Divisaodedados.py"
    ```

---

## 🌲 Treinamento do Modelo

=== "Código de treinamento"

    ```python
    --8<-- "docs/Arvore/Treinamentodomodelo.py"
    ```

---

## 📈 Avaliação do Modelo

!!! example "Resultado e visualização da árvore"

    O desempenho do modelo foi avaliado com acurácia e matriz de confusão. A árvore gerada também foi visualizada com `plot_tree()`.

=== "Decision Tree"

    ```python
    --8<-- "docs/Arvore/Avaliacaodomodelo.py"
    ```

=== "Código da Avaliação"

    ```python
    --8<-- "docs/Arvore/Avaliacaodomodelo.py"
    ```

---

## ✅ Conclusão

Este projeto demonstrou o uso de árvores de decisão na predição de diabetes. O modelo é interpretável e pode ser melhorado com ajustes de parâmetros ou ensemble methods como Random Forest.
