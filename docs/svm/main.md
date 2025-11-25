=== "code"
   ``` python exec="off"
   --8<-- "./docs/svm/svm.py"
   ```

=== "SVM"
   ![alt text](Image/Figure_6.png)
------------------------------------------------------------------------------------------

# Projeto: SVM com Kernel RBF — Explicação Completa do Código

Este documento apresenta uma explicação organizada do código utilizado para treinar um **SVM com Kernel RBF**, seguindo os critérios exigidos pelo Projeto Integrador.

---

## 1. Exploração dos Dados

O código inicia com o carregamento do arquivo **Testing.csv**, obtido preferencialmente do GitHub. Caso não seja possível, ele tenta automaticamente carregar versões locais do arquivo. Isso garante resiliência e confiabilidade no processo de leitura.

Após o carregamento, o script:

- Seleciona apenas **colunas numéricas**, necessárias para o treino do SVM.
- Utiliza as **duas primeiras variáveis numéricas** como *features* (X).
- Usa a **última coluna** como *target* (y).
- Verifica se o *target* possui **exatamente duas classes**, condição obrigatória para o SVM binário.

Essa etapa permite compreender a composição do dataset e garante que ele esteja estruturado corretamente para os passos seguintes.

---

## 2. Pré-processamento

Nesta etapa, o script realiza:

### Seleção de variáveis numéricas  
Garante que apenas dados contínuos sejam utilizados.

### Conversão dos rótulos  
Os valores do *target* são convertidos para **{-1, +1}**, conforme a formulação matemática do SVM.

### Verificações de integridade  
Confirma que há pelo menos duas features numéricas e que a classificação é binária.

### Normalização  
O código não normaliza explicitamente as features (ex.: StandardScaler), mas o Kernel RBF suaviza parcialmente essa necessidade.  
**Sugestão de melhoria:** adicionar normalização.

---

## 3. Divisão dos Dados

O código **não divide** o dataset em treino e teste.  
Todo o conjunto é utilizado para ajustar o modelo e gerar a figura da fronteira de decisão.

Embora isso funcione para fins didáticos e visualização, recomenda-se incluir divisão treino/teste em aplicações reais.

---

## 4. Treinamento do Modelo — SVM Dual com Kernel RBF

O treinamento é feito **manualmente**, implementando a forma dual do SVM.

### 4.1 Kernel RBF
A função de kernel gaussiano é definida como:

\[
K(x_i, x_j) = e^{-\frac{\|x_i - x_j\|^2}{2\sigma^2}}
\]

### 4.2 Construção da Matriz Kernel
A matriz K é calculada comparando cada par de observações, gerando um mapa de similaridade.

### 4.3 Otimização Dual
O problema dual do SVM é resolvido com restrições:

- \(0 \le \alpha_i \le C\)
- \(\sum \alpha_i y_i = 0\)

Utiliza-se `scipy.optimize.minimize` com o método **SLSQP**.

###  4.4 Vetores de Suporte
Os pontos com \(\alpha > 1e^{-6}\) são identificados como **support vectors**, elementos essenciais para definir a fronteira.

###  4.5 Cálculo do Bias (b)
O termo **b** é calculado usando a média dos erros nos vetores de suporte.

---

##  5. Avaliação do Modelo

O modelo é avaliado de forma **visual** através de:

- **Regiões de decisão coloridas**  
- **Fronteira de decisão pontilhada (linha Z=0)**  
- **Plot das classes -1 e +1**  
- **Destacando os vetores de suporte** com círculos sem preenchimento  

Embora não utilize métricas numéricas, o gráfico demonstra claramente a separação não linear obtida pelo kernel RBF.

**Sugestão:** incluir acurácia, F1-score e matriz de confusão para avaliação quantitativa (treino vs teste).

---

##  6. Relatório Final — Conclusões e Melhorias

O código fornece:

- A implementação completa do SVM Dual com Kernel RBF.
- Identificação dos vetores de suporte.
- Uma visualização clara da fronteira de decisão.
- Funções de decisão e predição.

### Possíveis melhorias para o Projeto Integrador:

1. Adicionar estatísticas descritivas (média, mediana, histograma).
2. Realizar normalização das features.
3. Inserir divisão treino/teste.
4. Testar diferentes valores de _C_ e _sigma_.
5. Implementar métricas numéricas para avaliar performance.
6. Comparar com o SVM do Scikit-Learn.
