=== "output"
    ``` python exec="on" html="1"
    --8<-- "./docs/K-mean/kmean.py"
    ```

=== "code"
    ``` python exec="off"
    --8<-- "./docs/K-mean/kmean.py"
    ```

## 1) Exploração dos dados

* **Leitura e natureza do conjunto**: o script lê um CSV remoto para um `DataFrame` (`pd.read_csv(...)`). Em seguida, **seleciona apenas as colunas numéricas** (`select_dtypes(include=[np.number])`) e **remove linhas com valores ausentes** nessas colunas (`dropna()`).
* **Seleção de features para visualização**: se houver **duas ou mais** colunas numéricas, o código **pega as duas primeiras** (`iloc[:, :2]`) e as converte para `numpy` para formar `X`. Se houver **apenas uma** coluna numérica, ele **duplica** essa coluna para construir um plano 2D e permitir o gráfico de dispersão.
* **Visualização**: cria uma figura (`plt.figure(...)`) e, após o agrupamento, **plota um scatter** dos pontos coloridos pelos rótulos de cluster e marca os **centróides** com um *asterisco* vermelho.
* **Estatísticas descritivas**: **não são calculadas** no script (não há `describe()`, médias, desvios, histogramas etc.).

## 2) Pré-processamento

* **Tratamento de ausentes**: o código faz `dropna()` nas colunas numéricas selecionadas, removendo linhas com NaN antes do modelo.
* **Normalização/Escala**: **não há normalização/padronização**; as features são usadas no **escala original**.
* **Outros passos** (remoção de outliers, codificação categórica etc.): **não são realizados**.

## 3) Divisão dos dados (treino e teste)

* **Não ocorre divisão**. O script **não** cria conjuntos de treino e teste; todo o `X` é utilizado diretamente no ajuste do algoritmo. (Isso é coerente com o uso atual do script, que faz **agrupamento** não supervisionado.)

## 4) Treinamento do modelo

* **Algoritmo implementado**: o script **treina K-Means** com `n_clusters=3`, inicialização `k-means++`, `max_iter=100` e `random_state=42`.
* **Ajuste e rótulos**: `fit_predict(X)` executa o agrupamento sobre **todas as observações** em `X` e retorna `labels`, que indicam a qual **cluster (0, 1, 2)** cada ponto pertence.

## 5) Avaliação do modelo

* **Avaliação visual**: o gráfico resultante mostra a **distribuição dos pontos por cluster** e a posição dos **centróides**. Essa é a base de avaliação presente no script.
* **Métricas numéricas**: há linhas **comentadas** para imprimir **centróides** e **inércia (WCSS)**; como estão comentadas, **nenhuma métrica é exibida**. Não há cálculo de outras métricas (ex.: *silhouette*).

## 6) Relatório final (processo, resultados, possíveis melhorias)

* **Processo implementado**:

  1. leitura do CSV e filtragem de colunas numéricas;
  2. remoção de ausentes;
  3. escolha de **duas features** (ou duplicação da única feature) para visualização 2D;
  4. treinamento do **K-Means (k=3)**;
  5. geração de **gráfico de dispersão** com rótulos e centróides;
  6. salvamento em **SVG** para *buffer* e **impressão** do conteúdo SVG no *stdout*.
* **Resultados gerados**: um **SVG** contendo o **gráfico de clusters** com três grupos e seus **centróides**.
* **Aspectos não contemplados no script** (apenas descrição): **estatísticas descritivas**, **normalização**, **divisão treino-teste**, **implementação de KNN** e **métricas de desempenho supervisionadas** **não** estão presentes no código atual.
