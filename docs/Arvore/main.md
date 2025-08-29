!!! example "Explicação da base escolhida e código de exploração"

O diabetes é uma das doenças crônicas mais comuns no mundo e pode causar complicações graves de saúde se não for diagnosticado e tratado corretamente. O dataset utilizado neste projeto é composto por informações clínicas de pacientes, como número de gestações, nível de glicose, pressão arterial, índice de massa corporal (IMC), nível de insulina e idade.

O objetivo é construir um modelo de classificação supervisionada que consiga prever, com base nesses atributos, se um paciente tem ou não diabetes (variável alvo Outcome: 0 = não diabético, 1 = diabético).

=== "Code"

    ```python
    --8<-- "docs/Arvore/Exploracaodedados.py"
    ```

!!! example "Explicação dos processos realizados no pré-processamento"

Na etapa de pré-processamento, foram identificados valores inconsistentes em algumas colunas, como Glucose, BloodPressure, SkinThickness, Insulin e BMI, que apresentavam valores iguais a zero — algo biologicamente impossível.
Esses valores foram tratados substituindo-os pela mediana de cada variável.

Após essa correção, foi aplicada a normalização com StandardScaler, garantindo que todas as variáveis fiquem na mesma escala (média 0 e desvio 1), o que ajuda no desempenho de vários algoritmos de machine learning.

=== "Code"

    ```python
    --8<-- "docs/Arvore/Preprocessamento.py"
    ```

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/Arvore/Treinamentodomodelo.py"
    ```



=== "Code"

    ```python
    --8<-- "docs/Arvore/Divisaodedados.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/Arvore/Treinamentodomodelo.py"
    ```


!!! example "Breast Cancer Dataset"

=== "decision tree"

    ```python exec="1" html="true"
    --8<-- "docs/Arvore/Avaliacaodomodelo.py"
    ```

=== "code"

    ```python exec="0"
    --8<-- "docs/Arvore/Avaliacaodomodelo.py"
    ```

