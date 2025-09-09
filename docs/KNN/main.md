``` python exec="on" html="1"
--8<-- "./docs/KNN/KNN.py"
```

KNN com o meu CSV: explicação do pipeline e dos resultados

O que o código faz
1) Carrega o CSV Testing.csv do repositório.
2) Define a variável alvo (Outcome, ou a última coluna caso Outcome não exista).
3) Prepara os dados: converte variáveis categóricas em colunas dummies (one hot), preenche ausentes com a mediana e mantém apenas valores numéricos para o modelo.
4) Divide em treino e teste com estratificação para preservar a proporção das classes.
5) Padroniza as features com StandardScaler para que todas fiquem na mesma escala, o que é essencial para KNN.
6) Treina um KNN com k = 3 usando os dados padronizados de treino.
7) Avalia no conjunto de teste e calcula as métricas.
8) Mostra duas figuras:
   • Matriz de confusão com contagens.
   • Fronteira de decisão em 2D usando PCA apenas para visualização.

Como os dados se relacionam com o modelo
• O KNN decide a classe de cada amostra olhando para os vizinhos mais próximos no espaço das features.
• Como as features foram padronizadas, cada coluna contribui de forma equilibrada para a distância.
• O gráfico de PCA 2D comprime todas as features em duas componentes principais só para visualizar. Para esse gráfico é treinado um KNN separado nas duas componentes, apenas para desenhar as regiões. As métricas reportadas vêm do KNN treinado com todas as features padronizadas.

O que é KNN em duas linhas
• KNN é um método baseado em instâncias: ele não aprende uma fronteira paramétrica, apenas armazena os dados de treino e classifica por voto dos k vizinhos mais próximos.
• A distância padrão é a euclidiana, mas é possível usar Manhattan e outras. O valor de k controla a suavidade da decisão.

Resultados principais deste experimento
• Conjunto de teste: 62 amostras
• Matriz de confusão (real × predito):
  – Verdadeiro negativo: 36
  – Falso positivo: 7
  – Falso negativo: 10
  – Verdadeiro positivo: 9
• Métricas da classe positiva:
  – Accuracy geral ≈ 0,726
  – Precisão ≈ 0,562
  – Recall ≈ 0,474
  – F1 score ≈ 0,514

Leitura rápida
O modelo acerta melhor a classe 0 e perde sensibilidade na classe 1, com mais falsos negativos. Isso acontece quando as classes estão desbalanceadas e quando há sobreposição entre elas.


• A diagonal mostra os acertos: 36 para a classe 0 e 9 para a classe 1.
• Fora da diagonal estão os erros:
  – 7 amostras da classe 0 viraram 1 (falsos positivos).
  – 10 amostras da classe 1 viraram 0 (falsos negativos).
• Se o foco for detectar a classe 1, importa reduzir falsos negativos e aumentar o recall da classe 1.

Melhora do código e do modelo
1) Escolha de k e de distância
   – Fazer busca em grade com validação cruzada para k em {3, 5, 7, 9, 11} e distância euclidiana ou Manhattan.
   – Usar weights='distance' pode ajudar quando a vizinhança é heterogênea.
2) Ajuste de decisão
   – Usar predict_proba e alterar o limiar da classe positiva de 0,50 para um valor que aumente o recall quando esse for o objetivo.
3) Balanceamento
   – Testar SMOTE ou subamostragem para balancear as classes no treino.
   – Manter divisão estratificada.
4) Limpeza de dados
   – Tratar outliers e padronizar sempre após a divisão em treino e teste, como já foi feito.
   – Verificar colunas quase constantes ou muito correlacionadas que pouco agregam informação.
5) Métricas adicionais
   – Incluir curva ROC e AUC e também curva Precisão Recall quando a classe positiva for rara.
   – Mostrar matriz de confusão em percentuais além de contagens.
6) Visualização
   – Lembrar que o gráfico com PCA é apenas uma projeção para 2D; ele ajuda a ver a sobreposição, mas não reflete toda a informação do modelo final.

Resumo em uma frase
Com os dados padronizados do CSV, o KNN com k igual a 3 alcançou cerca de 0,73 de accuracy, acerta bem a classe 0, mas precisa de ajustes para melhorar o recall da classe 1; tuning de k e distância, uso de pesos por distância, ajuste de limiar e balanceamento no treino tendem a melhorar esse comportamento.
