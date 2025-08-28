O dataset contém 308 registros e 9 variáveis relacionadas a exames médicos e características pessoais, sendo usado para prever a presença ou não de diabetes.

As variáveis são:
Pregnancies: número de gestações.
Glucose: nível de glicose no sangue.
BloodPressure: pressão arterial diastólica (mm Hg).
SkinThickness: espessura da dobra cutânea (mm).
Insulin: nível de insulina no sangue.
BMI: índice de massa corporal.
DiabetesPedigreeFunction: função que estima risco genético de diabetes.
Age: idade.
Outcome: variável alvo (0 = não diabético, 1 = diabético).

Estatísticas Descritivas
Idade varia de 21 a 70 anos, com média de ~34.
Glicose tem média de ~120, variando entre 0 e 199 (valores 0 sugerem dados faltantes codificados como 0).
Pressão Arterial, Espessura da Pele, Insulina e IMC também possuem registros com valor 0, o que provavelmente representa ausência de medição.
IMC médio é ~31.9 (sobrepeso).
A variável alvo (Outcome) mostra que cerca de 30% dos pacientes são diabéticos.

Visualizações
Distribuição da glicose: mostra uma concentração em torno de 100–140, com alguns outliers altos.

Distribuição da idade: maioria entre 20 e 40 anos.

Distribuição do desfecho (Outcome): a maior parte dos pacientes não é diabética (classe desbalanceada).

Neste projeto utilizei um modelo de Árvore de Decisão para prever se um paciente possui diabetes com base em características clínicas. O conjunto de dados contém informações como número de gestações, nível de glicose, pressão arterial, espessura da pele, insulina, índice de massa corporal (BMI), histórico familiar (Diabetes Pedigree Function) e idade. A variável alvo é Outcome, que indica se o paciente é diabético (1) ou não (0).

O processo foi dividido em etapas:

Carregamento da base: os dados foram lidos e a variável Outcome foi definida como alvo do modelo.

Divisão em treino e teste: os dados foram separados em 80% para treinamento e 20% para teste, garantindo a avaliação imparcial do modelo.

Treinamento do modelo: O algoritmo de árvore de decisão, limitando a profundidade máxima para evitar sobreajuste (overfitting).

Avaliação: a acurácia do modelo foi calculada, junto com métricas de precisão, recall, F1-score e a matriz de confusão, que mostram a qualidade das previsões.

A árvore de decisão resultante é uma representação gráfica do processo de classificação. Em cada nó, o modelo faz uma pergunta baseada em uma variável (por exemplo, “nível de glicose > 120?”). Dependendo da resposta, o algoritmo segue por um caminho até chegar a uma folha, que representa a previsão final (diabético ou não diabético). As cores dos nós ajudam a visualizar a predominância de cada classe, e as folhas indicam a decisão do modelo.

Em resumo, a árvore de decisão nos permite não apenas classificar novos pacientes, mas também interpretar facilmente quais fatores clínicos têm maior peso no diagnóstico de diabetes, tornando o modelo útil tanto para previsão quanto para análise exploratória dos dados.

``` python exec="on" html="0"
--8<-- "./docs/Arvore/Arvore.py"
```