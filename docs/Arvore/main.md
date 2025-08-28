!!! example "Explicação da base escolhida e codigo de exploração"

O câncer de mama é o câncer mais comum entre as mulheres do mundo. É responsável por 25% de todos os casos de câncer e afetou mais de 2,1 milhões de pessoas apenas em 2015. Começa quando as células da mama começam a crescer fora de controle. Essas células geralmente formam tumores que podem ser vistos via raios-X ou sentidos como nódulos na área da mama.

O principal desafio contra sua detecção é como classificar os tumores em malignos (cancerosos) ou benignos (não cancerosos), o intuito dessa entrega é criar um modelo que preveja a variavel target, classificada em tumores malignos ou benignos.

=== "Code"

```python
--8<-- "docs/Arvore/Exploracaodedados.py"
```

Pré-Processamento
!!! example "Explicação dos processos realizados no pré-processamento"

Na etapa de pré-processamento, os dados do dataset de cancer de mama passaram por um processo de limpeza de dados, tratamento de valores ausentes e label encoding. Colunas irrelevantes para o modelo foram retiradas por exemplo a coluna ['id'] , foi realizada a imputação com mediana de valores ausentes nas features concavity_worts e concavity points_worst e conversão de caracteres para números com labelEncoder na variavel target diagnostico.

=== "Code"

```python
--8<-- "docs/Arvore/Preprocessamento.py"
```

=== "Resultado"

```python exec="on" html="0"
--8<-- "docs/Arvore/Treinamentodomodelo.py"
```

Divisão de Dados
!!! example "Explicação da etapa de divisão de dados"

Na etapa de divisão de dados do dataset, eles foram separados em uma proporção de 30% teste e 70% treino.

=== "Code"

```python
--8<-- "docs/Arvore/Divisaodedados.py"
```

Treinamento do modelo
!!! example "ETAPA I:" Na etapa I o modelo foi testado e treinado usando uma proporção de 20% teste 80% treino, com essas porcentagens o modelo apresentou 93% de acuracia.

!!! example "ETAPA II:" NA etapa II o modelo foi testado e treinado usando uma proporção de 30% teste e 70% treino, com essas porcentagens o modelo apresentou 90% de acuracia.

=== "Code"

```python
--8<-- "docs/Arvore/Treinamentodomodelo.py"
```

Avaliação do Modelo Final
Após realizar o treinamento do modelo em diferentes divisões de treino e teste, foi feita a avaliação final utilizando o algoritmo de árvore de decisão.

O objetivo dessa etapa foi verificar como o modelo se comporta diante dos dados de teste, analisando métricas de desempenho como acurácia e a complexidade da árvore.

Durante a avaliação, percebi que a árvore gerada estava ficando relativamente pequena, o que podia indicar que o modelo está simplificando demais os padrões dos dados, porém corrigindo a quantidade de dados que estavam sendo usados, a árvore pareceu mais coesa.

Ainda assim, os resultados mostraram uma acurácia razoável, o que significa que o modelo conseguiu classificar corretamente a maioria dos casos entre tumores benignos e malignos.

!!! example "Breast Cancer Dataset"

=== "decision tree"

```python exec="1" html="true"
--8<-- "docs/Arvore/Avaliacaodomodelo.py"
```

=== "code"

```python exec="0"
--8<-- "docs/Arvore/Avaliacaodomodelo.py"
```

Relatório Final
!!! example "Resumo do Projeto"

Este projeto teve como objetivo aplicar técnicas de Machine Learning para criar um modelo capaz de prever se um tumor de mama é benigno ou maligno, utilizando o dataset Breast Cancer Wisconsin (Diagnostic).

As etapas seguidas foram:

Exploração de dados - Pré-processamento - Divisão de dados – separação em treino e teste - Treinamento do modelo - Avaliação do modelo – análise do desempenho final com base na acurácia e na estrutura da árvore.

!!! success "Resultados Obtidos" Acurácia variando entre 90% e 93%, dependendo da proporção de treino/teste utilizada.

O modelo conseguiu generalizar relativamente bem, mas ainda apresenta limitações na complexidade da árvore.

!!! tip "Conclusão" Mesmo com limitações, o projeto cumpriu seu objetivo: desenvolver um modelo de classificação supervisionada e aplicar todo o fluxo de pré-processamento, treino e avaliação, consolidando o meu aprendizado sobre o processo de Machine Learning.