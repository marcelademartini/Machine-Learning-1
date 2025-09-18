=== "output"
   ``` python exec="on" html="1"
   --8<-- "./docs/KNN/KNN.py"
   ```

=== "code"
   ``` python exec="off"
   --8<-- "./docs/KNN/KNN.py"
   ```

----------------------------------------------------------------------------------------------------------------------------

## 1) Exploração dos Dados

* Leitura do conjunto de dados: o CSV é carregado de uma URL do GitHub para o DataFrame df:

* df = pd.read_csv('https://raw.githubusercontent.com/.../Testing.csv').

* Natureza do problema (supervisionado): o código define automaticamente a coluna-alvo para classificação:

* target = 'Outcome' se essa coluna existir; caso contrário, usa a última coluna do CSV (df.columns[-1]).

* Separação inicial de variáveis:

* **X_raw = df.drop(columns=[target]) (preditoras).**

* **y = df[target] (alvo).**

* Tipos de dados:

* Preditoras categóricas são transformadas em dummies com pd.get_dummies(..., drop_first=True), o que cria colunas binárias para categorias.

* Se o alvo não for numérico, ele é codificado em inteiros com pd.factorize.

* Visualizações/estatísticas descritivas: nesta etapa não há gráficos exploratórios nem estatísticas como média, desvio padrão, etc. A exploração aqui se resume à leitura do CSV, identificação da coluna-alvo e preparação de tipos (numérico/categórico).

## 2) Pré-processamento

* **Tratamento de ausentes (NaN):**

**X = X.fillna(X.median(numeric_only=True)):** preenche valores faltantes nas colunas numéricas com a mediana de cada coluna.

* **Codificação de categóricas:**

**pd.get_dummies(X_raw, drop_first=True):** converte todas as colunas categóricas de X_raw para variáveis indicadoras, descartando a primeira categoria (evita redundância).

* **Padronização (normalização z-score):**

* StandardScaler() é ajustado em X_train e aplicado em X_train/X_test, produzindo X_train_s e X_test_s. Isso centraliza (média 0) e escala (desvio 1) as features, o que é importante para KNN.

## 3) Divisão dos Dados

* Hold-out 80/20:

* train_test_split(..., test_size=0.2, random_state=42, stratify=...).

* **Estratificação:**

* Se houver mais de uma classe no alvo, o split é estratificado para manter as proporções de classes em treino e teste.

* **Reprodutibilidade:**

* random_state=42 garante que a mesma divisão seja reproduzível.

## 4) Treinamento do Modelo (KNN)

* Algoritmo: KNeighborsClassifier com n_neighbors=k, onde k = 3.

* **Treinamento:**

* knn.fit(X_train_s, y_train) usa as features padronizadas de treino.

* **Predição:**

* y_pred = knn.predict(X_test_s) gera as classes previstas para o conjunto de teste.

## 5) Avaliação do Modelo

* **Métricas impressas:**

* accuracy_score(y_test, y_pred) (acurácia) é calculada e armazenada em acc.

* classification_report(y_test, y_pred, digits=3) é impresso no console com precision, recall, f1-score e support por classe, além das médias.

* **Matriz de confusão (figura):**

* confusion_matrix(y_test, y_pred) gera a matriz cm.

* Em seguida, é plotada com plt.imshow(cm, ...), adiciona-se plt.colorbar() e os números das células são sobrepostos com plt.text(...).

* A figura é exportada como SVG por print_svg_current_fig(), que salva no buffer (StringIO) e imprime o SVG no stdout (útil para ambientes que capturam a saída).

Visualização da fronteira de decisão em 2D (PCA):

* Se X_train tiver ≥ 2 features, aplica-se PCA para reduzir X_train_s/X_test_s a 2 componentes: X_train_2d, X_test_2d.

* Treina-se um KNN separado para visualização (knn_viz) neste espaço 2D.

* Cria-se uma malha (np.meshgrid) e plota-se plt.contourf(...) com as regiões de decisão do KNN em 2D.

* Os pontos de treino (círculos) e teste (xis) são sobrepostos, coloridos pelas classes verdadeiras.

* Esta figura também é exportada como SVG via print_svg_current_fig().

## 6) Relatório Final (documentação do que o código produz)

* **Processo documentado pelo código:**

* Entrada: leitura do CSV remoto e definição automática do alvo.

* Pré-processamento: dummies para categóricas, fatorização do alvo se necessário, imputação por mediana para NaN e padronização via StandardScaler.

* Divisão: treino/teste 80/20 com estratificação quando aplicável.

* Modelo: KNN com k=3, treinado em dados padronizados.

* **Saídas:**

* Texto: relatório de classificação (precision, recall, f1, support) e a acurácia (em acc).

* Figuras (SVG, impressas no stdout):

* Matriz de confusão do conjunto de teste.

* Fronteira de decisão em 2D após redução por PCA, com pontos de treino e teste.

* **Resultados obtidos:**

* O console exibirá o classification_report com métricas por classe e médias; a acurácia está disponível na variável acc.

* Duas figuras SVG são geradas na saída: (i) Matriz de confusão e (ii) Fronteira de decisão (PCA 2D).

*Possíveis melhorias (nota descritiva): não fazem parte do código atual; portanto, o relatório final se limita às etapas e produtos acima executados pelo script.