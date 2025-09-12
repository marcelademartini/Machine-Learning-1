# TESTE ARVORE DE DECISÃO


``` python exec="on" html="1"
--8<-- "./docs/Arvore/Arvore.py"
```

``` python exec="on" html="1"
--8<-- "./docs/Arvore/arv.py"
```

# Classificação de Diabetes usando Árvore de Decisão

## 1) Exploração dos Dados

Natureza dos dados: a base tem 2.460 linhas e 9 colunas, com a coluna alvo Outcome (classificação binária: 0/1). As features incluem: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

Estrutura e tipos: gerei um CSV com tipo de dado e contagem de nulos por coluna: meta_colunas.csv.

Estatísticas descritivas: calculei média, desvio padrão, mínimos e quartis de todas as colunas numéricas: descritivas_numericas.csv.

Insight rápido: variáveis como Glucose, BMI e Age costumam ter relação forte com o desfecho de diabetes; isso se reflete nas importâncias do modelo.


Este projeto tem como objetivo desenvolver um modelo de **Machine Learning** para **classificar pacientes** entre **diabéticos** e **não diabéticos**, utilizando uma **Árvore de Decisão**.  
O modelo foi construído com base no **Pima Indians Diabetes Dataset**.

O fluxo do projeto é dividido em etapas:
- Exploração e análise dos dados;
- Pré-processamento e limpeza;
- Divisão do conjunto de dados;
- Treinamento do modelo;
- Avaliação de desempenho;
- Discussão de melhorias futuras.

Valores ausentes: para colunas numéricas, preenchi eventuais ausências com a mediana; para categóricas (não há na base final), seria a moda.

Codificação: como todas as features já são numéricas e a variável alvo é binária, não foi necessária one-hot. Mantive um LabelEncoder genérico caso a base traga strings.

---

## 2. Exploração dos Dados 

A base de dados contém **308 registros** e **9 atributos**, sendo um deles a variável alvo (**Outcome**).

| Coluna                     | Descrição                                   | Tipo   |
|---------------------------|-------------------------------------------|--------|
| **Pregnancies**           | Número de vezes que a paciente engravidou | int    |
| **Glucose**               | Concentração de glicose no plasma         | int    |
| **BloodPressure**         | Pressão arterial diastólica (mm Hg)       | int    |
| **SkinThickness**         | Espessura da pele no tríceps (mm)         | int    |
| **Insulin**               | Nível de insulina sérica (µU/ml)          | int    |
| **BMI**                   | Índice de Massa Corporal                 | float  |
| **DiabetesPedigreeFunction** | Probabilidade genética de diabetes     | float  |
| **Age**                   | Idade da paciente (anos)                 | int    |
| **Outcome**               | Diagnóstico (0 = não diabético, 1 = diabético) | int |

**Principais insights iniciais:**
- Média da glicose: **~120 mg/dL**
- IMC médio: **~31,8**
- Idade média: **~33 anos**
- Proporção de pacientes diabéticos: **~30%**

---

### 3) Divisão dos Dados

## Split treino/teste: usei train_test_split com test_size=0.2 e random_state=42.

Estratificação: quando possível, estratifiquei por y para manter a proporção de classes no conjunto de teste, garantindo avaliação mais justa.


### 4) Treinamento do Modelo

Algoritmo: DecisionTreeClassifier(random_state=42).

Ajuste básico: sem poda inicial para mostrar a árvore “plena”. Em contexto de produção, recomenda-se ajustar hiperparâmetros como max_depth, min_samples_split e min_samples_leaf para reduzir overfitting.

Treino: clf.fit(X_train, y_train).

### 5) Relatório Final 

Objetivo. Construímos um classificador de árvore de decisão para prever o desfecho binário Outcome (0/1) a partir de variáveis clínicas relacionadas a diabetes.
Base e exploração. O conjunto Training.csv contém 2.460 observações e 9 colunas. As features são numéricas (gravidez, glicose, pressão arterial, espessura de pele, insulina, IMC, pedigree de diabetes e idade). Estatísticas descritivas indicam ampla variação em Glucose, BMI e Age, o que sugere potencial discriminativo (ver arquivo de descritivas).
Pré-processamento. Tratei valores ausentes: mediana para numéricos (e moda para categóricos, se necessário). Como árvores não dependem de escala, não normalizei.
Divisão. Separei os dados em treino (80%) e teste (20%) com random_state=42; estratificação foi usada para preservar as proporções de classe.
Modelagem. Treinei um DecisionTreeClassifier básico (sem restrições de profundidade) para evidenciar a lógica de particionamento.
Avaliação. O relatório de classificação apresenta métricas elevadas para ambas as classes (precisão, recall e F1). A análise de importância das variáveis aponta maior contribuição de atributos clínicos esperados, como Glucose, BMI, Age e DiabetesPedigreeFunction.
Limitações e melhorias. A acurácia elevada pode indicar overfitting ou leakage em bases muito limpas/estruturadas. Recomenda-se: (i) aplicar poda (ajustar max_depth, min_samples_leaf, min_samples_split), (ii) realizar validação cruzada e busca em grade de hiperparâmetros, (iii) comparar com modelos de ensemble (Random Forest, Gradient Boosting) e lineares, (iv) inspecionar outliers/valores impossíveis (ex.: pressões/espessuras zero) e (v) avaliar estabilidade temporal se houver dados ao longo do tempo. Para comunicação, manter gráficos de matriz de confusão, importâncias e uma árvore podada para melhor legibilidade.

