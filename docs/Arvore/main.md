# TESTE ARVORE DE DECISÃO


``` python exec="on" html="1"
--8<-- "./docs/Arvore/Arvore.py"
```

# Classificação de Diabetes usando Árvore de Decisão

## 1. Introdução 


### Sobre a Diabetes 

A **diabetes mellitus** é uma doença metabólica caracterizada pelo aumento anormal dos níveis de **glicose no sangue** (hiperglicemia).  
Ela ocorre quando o organismo **não produz** ou **não utiliza corretamente** a **insulina**, hormônio responsável por controlar a entrada de glicose nas células para gerar energia.

Existem dois tipos principais:
- **Diabetes Tipo 1** → O corpo não produz insulina. É mais comum em jovens e requer tratamento com insulina externa.
- **Diabetes Tipo 2** → O corpo produz insulina, mas não a utiliza de forma eficaz (resistência à insulina). Está relacionada a fatores como **alimentação, sedentarismo e genética**.

O diagnóstico precoce e o acompanhamento médico são fundamentais para prevenir complicações, como doenças cardiovasculares, problemas renais e neuropatias.



Este projeto tem como objetivo desenvolver um modelo de **Machine Learning** para **classificar pacientes** entre **diabéticos** e **não diabéticos**, utilizando uma **Árvore de Decisão**.  
O modelo foi construído com base no **Pima Indians Diabetes Dataset**.

O fluxo do projeto é dividido em etapas:
- Exploração e análise dos dados;
- Pré-processamento e limpeza;
- Divisão do conjunto de dados;
- Treinamento do modelo;
- Avaliação de desempenho;
- Discussão de melhorias futuras.

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

## 3. Melhorias 

### a) Normalização 
- Escalar as variáveis para melhorar a performance do modelo:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
