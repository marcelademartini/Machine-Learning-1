def gerar_relatorio():
    texto = """
📑 RELATÓRIO FINAL

1. Exploração dos Dados:
   - O dataset contém informações de saúde para prever diabetes.
   - Foi identificada a presença de valores ausentes representados como "0".

2. Pré-processamento:
   - Valores inválidos foram substituídos por NaN e imputados pela mediana.
   - Os dados foram normalizados (Z-score).

3. Modelo:
   - Algoritmo utilizado: Árvore de Decisão.
   - Base dividida em treino e teste (80/20).

4. Avaliação:
   - Métricas: acurácia, matriz de confusão, relatório de classificação.
   - Resultados indicam desempenho consistente para detecção de diabetes.

5. Possíveis Melhorias:
   - Testar outros modelos (Random Forest, SVM, Redes Neurais).
   - Balanceamento de classes (SMOTE).
   - Cross-validation para melhor estimativa do desempenho.
    """
    print(texto)
