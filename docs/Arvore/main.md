Neste projeto utilizei um modelo de Árvore de Decisão para prever se um paciente possui diabetes com base em características clínicas. O conjunto de dados contém informações como número de gestações, nível de glicose, pressão arterial, espessura da pele, insulina, índice de massa corporal (BMI), histórico familiar (Diabetes Pedigree Function) e idade. A variável alvo é Outcome, que indica se o paciente é diabético (1) ou não (0).

O processo foi dividido em etapas:

Carregamento da base: os dados foram lidos e a variável Outcome foi definida como alvo do modelo.

Divisão em treino e teste: os dados foram separados em 80% para treinamento e 20% para teste, garantindo a avaliação imparcial do modelo.

Treinamento do modelo: O algoritmo de árvore de decisão, limitando a profundidade máxima para evitar sobreajuste (overfitting).

Avaliação: a acurácia do modelo foi calculada, junto com métricas de precisão, recall, F1-score e a matriz de confusão, que mostram a qualidade das previsões.

A árvore de decisão resultante é uma representação gráfica do processo de classificação. Em cada nó, o modelo faz uma pergunta baseada em uma variável (por exemplo, “nível de glicose > 120?”). Dependendo da resposta, o algoritmo segue por um caminho até chegar a uma folha, que representa a previsão final (diabético ou não diabético). As cores dos nós ajudam a visualizar a predominância de cada classe, e as folhas indicam a decisão do modelo.

Em resumo, a árvore de decisão nos permite não apenas classificar novos pacientes, mas também interpretar facilmente quais fatores clínicos têm maior peso no diagnóstico de diabetes, tornando o modelo útil tanto para previsão quanto para análise exploratória dos dados.