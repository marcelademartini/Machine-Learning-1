# TESTE ARVORE DE DECISÃO

=== "output"
    ``` python exec="on" html="1"
    --8<-- "./docs/Arvore/arvore1.py"
    ````


------------------------------------------------------------------------------------------------------------

=== "output"
    ``` python exec="on" html="1"
    --8<-- "./docs/Arvore/Arvore.py"
    ```


=== "code"
    ``` python exec="off"
    --8<-- "./docs/Arvore/Arvore.py"
    ```



---------------------------------------------------------------------------------------------------------------

=== "output"
    ``` python exec="on" html="1"
    --8<-- "./docs/Arvore/arv.py"
    ```

=== "code"
    ``` python exec="off" 
    --8<-- "./docs/Arvore/arv.py"
    ```


# Código 1

## 1) Exploração dos dados

* O código carrega o arquivo CSV hospedado no GitHub em df usando pd.read_csv(...).

* A natureza do conjunto é tratada de forma implícita: parte-se do princípio de que existe uma coluna alvo chamada Outcome e que as demais colunas são atributos preditores.

## 2) Pré processamento

* O código assume que as colunas de x são compatíveis com o modelo (por exemplo, numéricas ou já codificadas). 

## 3) Divisão dos dados

* As variáveis são separadas em preditores e alvo:

* x = df.drop(columns=['Outcome']) contém todas as colunas exceto a coluna alvo.

* y = df['Outcome'] contém a classe discreta a ser prevista.

* A divisão treino e teste é feita por train_test_split(x, y, test_size=0.2, random_state=42), reservando 20% dos dados para teste e fixando a semente aleatória em 42 para reprodutibilidade.

## 4) Treinamento do modelo

* Cria se um classificador de árvore de decisão com tree.DecisionTreeClassifier() usando os padrões da biblioteca (por exemplo, critério Gini e profundidade livre, a menos que o dataset limite).

* O ajuste é realizado com classifier.fit(x_train, y_train), aprendendo regras de decisão a partir das amostras de treinamento.

## 5) Avaliação do modelo

* O desempenho é calculado com classifier.score(x_test, y_test), que retorna a acurácia média no conjunto de teste. O valor é impresso com duas casas decimais por print(f"Accuracy: {accuracy:.2f}").

* A visualização da estrutura aprendida é feita por tree.plot_tree(classifier), desenhando os nós e divisões na figura previamente aberta com plt.figure(figsize=(12, 10)).

## 6) Relatório final e saída gráfica

* Para disponibilizar a figura em contexto HTML, o código cria um StringIO, salva a figura atual em SVG com plt.savefig(buffer, format="svg") e imprime o conteúdo do buffer com print(buffer.getvalue()).

* O resultado produzido pelo script é:

* a acurácia calculada no conjunto de teste, e

* a árvore de decisão renderizada como SVG, impressa diretamente na saída padrão, pronta para ser consumida por uma página que leia essa saída e exiba o SVG.

* accuracy_score é importado mas não é utilizado, pois a acurácia é obtida via classifier.score(...).

# Código 2

## 1) Exploração dos dados

* O script carrega a base diretamente da URL para o DataFrame df:

* df = pd.read_csv('https://raw.githubusercontent.com/.../Testing.csv')


* Há comentários indicando as funções de inspeção inicial: df.head() para amostra de linhas, df.dtypes para tipos por coluna, df.describe() para estatísticas descritivas e df.isna().sum() para contagem de ausentes.

* Nesta etapa o código não imprime nada por padrão, apenas aponta quais comandos usar para entender a natureza das variáveis. A base contém a coluna alvo Outcome, usada mais adiante.

## 2) Preprocessamento

* O bloco de preprocessamento está comentado. Ele mostra como:

* Codificar categorias com LabelEncoder caso exista alguma coluna categórica.

* Tratar ausentes usando a mediana numérica: df = df.fillna(df.median(numeric_only=True)).

* Como está escrito, o fluxo segue sem executar transformações. Ou seja, o modelo utilizará os dados como estão em df.

## 3) Divisão dos dados

* O código separa features e alvo:

* x = df.drop(columns=['Outcome'])
* y = df['Outcome']


* Em seguida, faz a partição treino e teste com proporção 80/20 e semente fixa para reprodutibilidade:

* x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


* Isso garante um conjunto de teste mantido para avaliação fora do treino.

## 4) Treinamento do modelo (Decision Tree)

* O classificador é criado com random_state=42 e hiperparâmetros padrão:

* classifier = tree.DecisionTreeClassifier(random_state=42)


* O ajuste do modelo ocorre com:

* classifier.fit(x_train, y_train)

## 5) Avaliação do modelo e visualização

* A acurácia é calculada usando classifier.score(x_test, y_test) e exibida com duas casas decimais:

* accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


* Observação descritiva: classifier.score em um classificador equivale à acurácia. O módulo também importou accuracy_score, mas a métrica é obtida via .score() no seu código.

* A árvore treinada é plotada:
---------------------------------------------------------------------------------------------------------
plt.figure(figsize=(12, 10))
tree.plot_tree(classifier)
---------------------------------------------------------------------------------------------------------

* O gráfico apresenta a estrutura de decisão aprendida, com informações padrão nos nós como gini, amostras, distribuição por classe e classe final.

* Para uso em páginas HTML, o gráfico é exportado como SVG para um buffer de texto e o conteúdo SVG é impresso:
---------------------------------------------------------------------------------------------------------
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
-------------------------------------------------------------------------------------------------------------

* Isso produz o markup SVG completo, próprio para incorporação em HTML.

## 6) Relatório final

* Com base no que o script produz, o relatório pode documentar:

* Dados e contexto: origem do Testing.csv e breve descrição das variáveis observadas na exploração.

* Metodologia: uso de Decision Tree para classificação com divisão 80/20.

* Resultados: valor de acurácia impresso pelo código e a visualização da árvore gerada em SVG.

* Discussão: leitura da estrutura da árvore a partir do diagrama, destacando caminhos de decisão e nós mais relevantes conforme o gráfico.

* O material gerado atende aos itens solicitados no template do projeto integrador: dados, método, treinamento, métrica de avaliação e figura da árvore para ilustração dos resultados.