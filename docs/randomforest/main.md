=== "output"
   ``` python exec="on" html="1"
   --8<-- "./docs/randomforest/randomforest.py"
   ```

=== "code"
   ``` python exec="off"
   --8<-- "./docs/randomforest/randomforest.py"
   ```

-----------------------------------------------------------------------------------------------------

# Random Forest – Classificação de Diabetes

## 1) Exploração dos Dados
- **Fonte:** `Testing.csv` (dados clínicos de pacientes).
- **Alvo:** `Outcome` (0 = sem diabetes, 1 = com diabetes).
- **Atributos típicos:** Glucose, BloodPressure, BMI, Age, Pregnancies, SkinThickness, Insulin, DiabetesPedigreeFunction.
- **Natureza:** variáveis numéricas contínuas; cada linha = um paciente.

> Observação: é um conjunto na linha do Pima Indians Diabetes, usado amplamente em tarefas de classificação binária.

---

## 2) Pré-processamento
- **Leitura robusta:** tentativa por URL com *retries*; se falhar, busca arquivos locais candidatos.
- **Conversão:** `pd.to_numeric(errors="coerce")` em todas as features.
- **Valores ausentes:** preenchimento pela **mediana** por coluna.
- **Alvo:** garantido como inteiro/categórico codificado.
- **Reprodutibilidade:** `random.seed(42)` e `np.random.seed(42)`.

---

## 3) Divisão dos Dados
- **Estratégia:** embaralhamento de índices + split manual.
- **Proporção:** 80% treino | 20% validação/teste.
- **Motivo:** avaliar o desempenho em registros nunca vistos.

---

## 4) Treinamento do Modelo (Implementação Própria)
- **Algoritmo:** Random Forest manual (conjunto de árvores de decisão).
- **Critério:** impureza de **Gini**.
- **Amostragem:** *bootstrap* por árvore; seleção aleatória de atributos por divisão.
- **Parâmetros principais:**
  - `n_estimators = 15`
  - `max_depth = 6`
  - `min_samples_split = 4`
  - `max_features = "sqrt"`

---

## 5) Avaliação do Modelo

### Acurácia
- **Validação/Teste:** **0,774** (77,4%).

### Relatório de Classificação
| Classe | Precision | Recall | F1-Score | Suporte |
|---|---:|---:|---:|---:|
| 0 (sem diabetes) | 0,900 | 0,783 | 0,837 | 46 |
| 1 (com diabetes) | 0,545 | 0,750 | 0,632 | 16 |
| **Acurácia** |  |  | **0,774** | 62 |
| **Média macro** | 0,723 | 0,766 | 0,734 | 62 |
| **Média ponderada** | 0,809 | 0,774 | 0,784 | 62 |

### Matriz de Confusão
|        | Previsto 0 | Previsto 1 |
|---|---:|---:|
| **Real 0** | 36 | 10 |
| **Real 1** | 4  | 12 |

- Acerta bem a classe 0 (precision alta).
- Para a classe 1, o **recall** é 0,75 (identifica 75% dos casos positivos), com mais falsos positivos.

### Importância das Variáveis (sklearn para interpretação)
| Variável | Importância |
|---|---:|
| Glucose | 0,274 |
| BMI | 0,148 |
| Age | 0,147 |
| DiabetesPedigreeFunction | 0,112 |
| Pregnancies | 0,109 |
| BloodPressure | 0,085 |
| SkinThickness | 0,066 |
| Insulin | 0,059 |

> A glicose é o preditor mais relevante, seguida por IMC e idade, o que é consistente com a literatura clínica.

---

## 6) Relatório Final e Melhorias

**Resumo:**  
O Random Forest implementado atingiu **77,4%** de acurácia, com bom desempenho geral e interpretação via importância de features. O modelo equilibra viés e variância graças ao comitê de árvores e à amostragem de atributos.

**Possíveis melhorias:**
- Balanceamento de classes (SMOTE, oversampling ou *class weights*).
- Explorar mais árvores e profundidade maior com validação cruzada.
- Avaliar **ROC/AUC**, precisão-recall e *threshold tuning* conforme o custo de erros.
- Testar padronização seletiva e engenharia de atributos (ex.: interações clínicas).
- Comparar com modelos baseline (Logística, KNN, XGBoost) e com *grid search*.
 kkkkkkkkk