=== "output"
   ``` python exec="on" html="1"
   --8<-- "./docs/Metrics/Metrics.py"
   ```

=== "code"
   ``` python exec="off"
   --8<-- "./docs/Metrics/Metrics.py"
   ```

------------------------------------------------------------------------------------------------------

=== "output"
   ``` python exec="on" html="1"
   --8<-- "./docs/Metrics/Metrics1.py"
   ```

=== "code"
   ``` python exec="off"
   --8<-- "./docs/Metrics/Metrics1.py"
   ```


------------------------------------------------------------------------------------------------------

# 1) KNN: passo a passo do meu código
## 1.1. Importação e leitura do CSV

* Eu importo as bibliotecas do Python que preciso e leio o arquivo Testing.csv. Se existir a coluna Outcome, uso como alvo. Caso contrário, pego a última coluna como alvo.

* Por que isso? Eu preciso de uma variável de saída para transformar o problema em classificação supervisionada.

### 1.2. Seleção de variáveis e tratamento de tipos

* Eu separo X (todas as colunas menos o alvo) e y (o alvo). Se houver colunas categóricas em X, transformo em dummies com get_dummies. Se y não for numérica, fatorizo para virar números inteiros.

* Por que isso? Modelos como KNN trabalham com números. Então eu deixo tudo numérico e consistente.

### 1.3. Tratamento de ausentes

* Eu preencho valores ausentes em X com a mediana das colunas numéricas.

* Por que isso? Evito perder linhas e mantenho a escala robusta a outliers.

### 1.4. Split treino e teste com estratificação

* Eu divido os dados em treino e teste usando train_test_split com test_size=0.2 e random_state=42. Se houver mais de uma classe, uso stratify=y para manter as proporções das classes nos dois conjuntos.

* Por que isso? Preciso medir desempenho em dados que o modelo não viu. A estratificação evita desbalancear as classes no split.

### 1.5. Padronização

* Eu padronizo X_train e X_test com StandardScaler (média 0 e desvio 1).

* Por que isso? KNN usa distâncias. Se as escalas forem diferentes, uma variável pode dominar a distância. Padronizar deixa tudo comparável.

### 1.6. Treino do KNN

* Eu escolho k = 3, crio KNeighborsClassifier(n_neighbors=3), ajusto com fit e gero previsões no teste com predict.

* Por que isso? KNN classifica um ponto olhando os vizinhos mais próximos. k=3 é um começo simples para experimentar.

### 1.7. Métricas e relatório

* Eu calculo a acurácia e imprimo classification_report, que traz precisão, recall e F1 por classe, além das médias.

* Como ler?

* Precisão: dos positivos que eu previ, quantos eram realmente positivos

* Recall: dos positivos reais, quantos eu acertei

* F1: balanço entre precisão e recall

### 1.8. Matriz de confusão com figura em SVG

* Eu calculo cm = confusion_matrix(y_test, y_pred), ploto com imshow, escrevo os números em cada célula e exporto a figura como SVG usando a função print_svg_current_fig().

* Por que isso? A matriz de confusão mostra onde o modelo acerta e onde se confunde entre as classes. Exportar em SVG mantém a imagem nítida no navegador e no GitPages.

### 1.9. Visualização 2D com PCA e fronteira de decisão

* Se houver pelo menos duas features, eu rodo um PCA para reduzir para 2 componentes e treino um KNN nesse espaço 2D apenas para visualização. Depois eu desenho a fronteira de decisão com contourf, marco os pontos de treino e teste, e exporto em SVG.

* Por que isso? Ver a fronteira de decisão ajuda a entender como o KNN está separando as classes depois da redução de dimensionalidade. É apenas ilustrativo.

## 2) K-Means: matriz de confusão usando a mesma lógica visual

* O K-Means é não supervisionado, então ele cria clusters sem saber a classe real. Mesmo assim, se o meu dataset tem uma coluna alvo (por exemplo, Outcome), eu posso comparar os clusters com essa coluna para inspecionar o alinhamento entre grupos encontrados e as classes reais.

### 2.1. O que eu faço antes de treinar

* Carrego o Testing.csv

* Confirmo que Outcome existe para poder comparar

* Seleciono apenas colunas numéricas

* Para visualização, uso as duas primeiras colunas numéricas. Se só tiver uma, eu duplico para formar um plano 2D

### 2.2. Treino do K-Means

* Eu rodo KMeans(n_clusters=3, init='k-means++', max_iter=100, random_state=42) e pego os rótulos de cluster com fit_predict.

* Por que 3 clusters? É um valor inicial para observar o comportamento. Posso variar para testar.

### 2.3. Matriz de confusão bruta: classes reais × clusters

* Eu calculo cm_raw = confusion_matrix(y_true, labels) e ploto do mesmo jeito que no KNN: imshow, números nas células e exportação SVG pela mesma função.

* Importante: esta matriz é bruta. Cluster 0 não quer dizer classe 0, porque os rótulos de cluster são arbitrários.

### 2.4. Remapeamento de clusters para classes

* Para tornar a leitura mais parecida com classificação, eu tento mapear os clusters para as classes reais. Eu testo todas as permutações possíveis de cluster para classe e escolho a que dá mais acertos. Com esse mapeamento, eu gero uma segunda matriz de confusão, agora “alinhada”.

* Por que isso ajuda? Ajuda a ler a matriz como se fosse uma predição de classe. Ainda é um modelo não supervisionado, mas o remapeamento torna a comparação mais clara.

### 2.5. Visualização 2D dos clusters

* Eu desenho um scatter plot dos pontos coloridos pelos rótulos de cluster e marco os centróides com estrelas. Exporto em SVG.

* Por que isso? Ver a distribuição no plano 2D ajuda a entender como os clusters ficaram.

## 3) Observações e limitações

* No KNN, a matriz de confusão é direta porque existem classes reais e previsões.

* No K-Means, a matriz de confusão precisa ser interpretada com cuidado, porque os rótulos de cluster são arbitrários. O remapeamento é uma etapa extra que melhora a comparação, mas não transforma o K-Means em um classificador supervisionado.

* Padronizar dados é essencial para KNN e geralmente ajuda bastante em métodos baseados em distância.

* No K-Means, escolher o número de clusters é uma decisão importante. Vale testar valores diferentes e usar métricas de clusterização como silhouette e Davies Bouldin.

## 4) Como reproduzir

* Coloque Testing.csv na mesma pasta dos notebooks ou scripts.

* Para o KNN, rode o script principal. Ele já:

* prepara dados

* treina o modelo

* imprime relatório de classificação

* gera e imprime a matriz de confusão

* exporta as figuras como SVG

* Para o K-Means, rode o script de clustering. Ele:

* gera a matriz de confusão bruta

* remapeia clusters para classes e gera a matriz alinhada

* cria o scatter com centróides

* exporta tudo como SVG

* Se eu quiser publicar no GitPages, os SVGs ficam nítidos e leves, então as figuras carregam rápido e com boa qualidade.

## 5) Ideias de extensão

* Ajustar k no KNN e comparar as métricas

* Usar validação cruzada para avaliar estabilidade

* Testar diferentes números de clusters e calcular silhouette

* Usar mais componentes no PCA e só reduzir para 2D na hora da visualização

* Comparar K-Means com outros métodos de clusterização como DBSCAN e GMM


# Comparação entre KNN e K-Means

## 1) Resultados do KNN
* **Matriz de confusão**  
  - 36 verdadeiros negativos  
  - 7 falsos positivos  
  - 10 falsos negativos  
  - 9 verdadeiros positivos  

* **Fronteira de decisão (PCA 2D)**  
  - As regiões de decisão mostram bastante sobreposição.  
  - Isso explica a acurácia de aproximadamente **72%**.  
  - O modelo acerta mais a classe **0**, mas erra bastante na classe **1**.  

---

## 2) Resultados do K-Means
* **Matriz de confusão bruta**  
  - Clusters não correspondem diretamente às classes.  
  - Há mistura significativa entre classe 0 e 1.  

* **Matriz de confusão remapeada**  
  - O alinhamento melhora os resultados, mas ainda há muitos falsos negativos (classe 1 confundida).  

* **Visualização dos clusters**  
  - Centrôides bem definidos.  
  - Porém, classes diferentes caem nos mesmos clusters.  

---

## 3) Comparação Final
* **KNN**
  - Supervisionado (usa rótulos).  
  - Acurácia ≈ 72%.  
  - Generaliza melhor e distingue padrões conhecidos.  

* **K-Means**
  - Não supervisionado (não usa rótulos).  
  - Clusters não refletem perfeitamente as classes reais.  
  - Mesmo remapeado, tem desempenho inferior.  

---

## 4) Conclusão
* O **KNN é o melhor modelo** para este dataset.  
* O K-Means é útil para explorar agrupamentos, mas não substitui um classificador supervisionado quando os rótulos estão disponíveis.  
