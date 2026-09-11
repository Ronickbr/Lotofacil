# OBJETIVO

Quero que você analise e melhore completamente o módulo de treinamento da IA responsável por analisar concursos históricos da Lotofácil.

O sistema atual utiliza `RandomForestClassifier` e exibe apenas uma métrica de "Acurácia média", atualmente em torno de `0.26`.

Essa abordagem precisa ser revisada, porque a Lotofácil sorteia 15 dezenas entre 25 e uma métrica simples de accuracy pode ser inadequada ou até enganosa para esse tipo de problema.

Sua tarefa é reestruturar o treinamento, validação e avaliação do modelo, preservando o sistema existente sempre que possível.

---

# 1. ANALISE O CÓDIGO ATUAL PRIMEIRO

Antes de alterar qualquer coisa:

- localize o módulo responsável pelo treinamento;
- identifique como os dados históricos são transformados em features;
- identifique o formato do `X`;
- identifique o formato do `y`;
- descubra se o problema está sendo tratado como:
  - classificação binária;
  - multilabel;
  - multiclass;
  - multioutput;
- identifique como `accuracy` está sendo calculada;
- verifique se usa:
  - `model.score()`;
  - `accuracy_score()`;
  - `cross_val_score()`;
  - outra métrica;
- verifique se existe data leakage;
- verifique se concursos futuros estão sendo usados indiretamente no treinamento;
- verifique se o treino e teste estão sendo embaralhados.

Não faça mudanças cegas.

Documente rapidamente o que encontrou antes de alterar a arquitetura.

---

# 2. CORRIGIR A ESTRATÉGIA DE VALIDAÇÃO

Concursos de loteria formam uma sequência temporal.

NÃO usar:

```python
train_test_split(..., shuffle=True)
```

ou qualquer estratégia que misture concursos futuros com concursos passados.

Implementar validação temporal.

Preferencialmente usar:

## Walk-Forward Validation

Exemplo conceitual:

```text
Concursos 1 até 1000
→ treina

Concurso 1001
→ testa

Concursos 1 até 1001
→ treina

Concurso 1002
→ testa

Concursos 1 até 1002
→ treina

Concurso 1003
→ testa
```

Também pode ser utilizado:

```python
TimeSeriesSplit
```

desde que não haja leakage temporal.

---

# 3. REPRESENTAÇÃO DO PROBLEMA

Cada concurso possui:

```text
25 dezenas possíveis
15 sorteadas
10 não sorteadas
```

Representar cada concurso como vetor binário:

```text
01 02 03 04 05 ... 25

1  0  1  1  0 ... 1
```

onde:

```text
1 = saiu
0 = não saiu
```

O modelo deve prever uma probabilidade independente/estimada para cada uma das 25 dezenas:

```text
P(01)
P(02)
P(03)
...
P(25)
```

Não quero apenas um vetor final de `0/1`.

Quero probabilidades.

---

# 4. FEATURE ENGINEERING

Crie features históricas para cada dezena.

Para cada número de 1 até 25, calcule ao menos:

## Frequência

```text
frequência últimos 5 concursos
frequência últimos 10
frequência últimos 20
frequência últimos 30
frequência últimos 50
frequência últimos 100
frequência últimos 200
frequência histórica
```

## Atraso

```text
quantos concursos desde última aparição
atraso médio histórico
atraso máximo
desvio padrão dos atrasos
percentil do atraso atual
```

## Recência

```text
saiu no concurso anterior
saiu há 2 concursos
saiu há 3 concursos
saiu há 4 concursos
saiu há 5 concursos
```

## Tendência

Calcular:

```text
média móvel curta
média móvel longa
diferença entre média curta e longa
z-score
inclinação da frequência
```

## Associação

Calcular:

```text
frequência conjunta com outras dezenas
força de associação entre pares
frequência em trincas
```

Evitar dimensionalidade excessiva sem tratamento.

Se necessário, utilizar agregações.

---

# 5. FEATURES DO CONCURSO

Além das features individuais das dezenas, considerar padrões globais dos concursos anteriores:

```text
quantidade de pares
quantidade de ímpares
soma total
média das dezenas
desvio padrão
quantidade de consecutivos
maior sequência consecutiva
quantidade por linha
quantidade por coluna
quantidade de primos
quantidade de Fibonacci
quantidade de múltiplos de 3
quantidade de dezenas repetidas do concurso anterior
```

Essas features devem ser calculadas APENAS com dados anteriores ao concurso previsto.

Nunca usar dados do próprio concurso-alvo.

---

# 6. MODELOS

Não utilizar apenas Random Forest.

Criar arquitetura comparativa com pelo menos:

```text
RandomForestClassifier
ExtraTreesClassifier
LogisticRegression
HistGradientBoostingClassifier
XGBoost, se já existir no projeto
LightGBM, se já existir no projeto
```

Não adicionar dependências externas desnecessariamente.

Se XGBoost ou LightGBM não estiverem instalados, mantenha-os opcionais.

---

# 7. ENSEMBLE

Criar um Ensemble.

Cada modelo deve retornar:

```text
probabilidade das 25 dezenas
```

Combinar as probabilidades:

```python
P_final = (
    w1 * P_random_forest +
    w2 * P_extra_trees +
    w3 * P_logistic +
    w4 * P_gradient_boosting
)
```

Inicialmente pode usar pesos iguais.

Depois permitir pesos definidos automaticamente conforme desempenho no backtest.

---

# 8. MÉTRICAS CORRETAS

Parar de usar apenas:

```text
accuracy
```

A interface deve mostrar métricas mais relevantes.

Calcular:

```text
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Log Loss
Brier Score
```

Mas a principal métrica operacional deve ser:

# ACERTOS ENTRE AS 15 DEZENAS MAIS PROVÁVEIS

Para cada concurso de teste:

1. gerar probabilidade para as 25 dezenas;
2. ordenar da maior para menor;
3. selecionar Top 15;
4. comparar com as 15 dezenas sorteadas.

Calcular:

```text
quantidade de acertos
```

Exemplo:

```text
Previsto:
01 02 03 05 07 08 10 11 13 14 16 18 20 23 25

Real:
01 02 04 05 07 08 10 12 13 14 16 18 21 23 25

Acertos:
12 / 15
```

---

# 9. MÉTRICAS DE ACERTOS

No backtest, calcular:

```text
média de acertos por concurso
mediana
desvio padrão
mínimo
máximo
```

E distribuição:

```text
8 acertos
9 acertos
10 acertos
11 acertos
12 acertos
13 acertos
14 acertos
15 acertos
```

Mostrar também:

```text
% com >= 10
% com >= 11
% com >= 12
% com >= 13
% com >= 14
% com 15
```

---

# 10. BASELINE ALEATÓRIO

Isso é obrigatório.

O modelo só pode ser considerado útil se superar estratégias aleatórias.

Criar baseline escolhendo 15 dezenas aleatoriamente.

Executar:

```text
mínimo 100.000 simulações
```

Preferencialmente:

```text
1.000.000
```

Calcular:

```text
média de acertos aleatórios
distribuição de acertos
percentis
```

A expectativa teórica é aproximadamente:

```text
9 acertos
```

porque:

```text
15 × 15 / 25 = 9
```

Comparar sempre:

```text
IA
VS
ALEATÓRIO
```

---

# 11. LIFT

Adicionar uma métrica:

```text
Lift sobre baseline
```

Exemplo:

```python
lift = (
    media_modelo - media_baseline
) / media_baseline
```

Exibir:

```text
Baseline aleatório: 9.00
Modelo: 9.47
Lift: +5.22%
```

Não afirmar que isso representa capacidade real de prever sorteios sem significância estatística.

---

# 12. TESTE DE SIGNIFICÂNCIA

Implementar avaliação estatística.

Comparar os resultados do modelo com baseline aleatório usando:

```text
bootstrap
teste de permutação
intervalo de confiança
```

Preferencialmente gerar:

```text
95% confidence interval
```

Exemplo:

```text
Média modelo:
9.42

IC 95%:
9.31 – 9.53
```

Se o desempenho não for significativamente superior ao acaso, informar claramente:

```text
"Não foi detectada vantagem estatisticamente significativa."
```

---

# 13. PREVENÇÃO DE OVERFITTING

Aplicar controles como:

```text
max_depth
min_samples_leaf
min_samples_split
max_features
regularização
early stopping quando aplicável
```

Não buscar apenas maximizar desempenho no histórico.

O objetivo principal é desempenho fora da amostra.

---

# 14. HIPERPARÂMETROS

Executar otimização somente dentro do conjunto de treinamento.

Pode usar:

```text
RandomizedSearchCV
Optuna, se já disponível
```

Mas respeitando divisão temporal.

NUNCA realizar otimização usando concursos futuros.

---

# 15. TOP-N

Além do Top 15, testar:

```text
Top 16
Top 17
Top 18
Top 19
Top 20
```

Esses grupos serão posteriormente utilizados por um módulo combinatório.

Exemplo:

```text
Top 18 dezenas candidatas
↓
algoritmo combinatório
↓
jogos de 15 dezenas
```

---

# 16. SCORE INDIVIDUAL DAS DEZENAS

Criar ranking final como:

```text
01  █████████████  72.4%
05  ████████████   69.8%
13  ███████████    68.1%
...
```

Cada dezena deve possuir:

```text
probabilidade do ensemble
posição no ranking
frequência curta
frequência longa
atraso
tendência
```

---

# 17. CALIBRAÇÃO DE PROBABILIDADE

Verificar se probabilidades estão calibradas.

Utilizar, se apropriado:

```python
CalibratedClassifierCV
```

Comparar:

```text
sigmoid
isotonic
```

Avaliar com:

```text
Brier Score
calibration curve
```

---

# 18. NÃO TRATAR PROBABILIDADE COMO CERTEZA

Importante:

O sistema deve evitar mensagens como:

```text
"Estas dezenas têm maior chance real de sair."
```

Preferir:

```text
"Estas dezenas receberam maior score segundo os padrões históricos analisados pelo modelo."
```

Porque sorteios independentes não se tornam previsíveis apenas com dados históricos.

---

# 19. PAINEL DE TREINAMENTO

Substituir a exibição simples:

```text
Modelo treinado com sucesso!
Acurácia média: 0.26
```

por algo semelhante a:

```text
MODELO TREINADO COM SUCESSO

Validação:
Walk-Forward

Concursos usados:
3.520

Concursos de backtest:
500

Média de acertos:
9.43 / 15

Baseline aleatório:
9.00 / 15

Lift:
+4.78%

Mediana:
9

Melhor resultado:
13 / 15

>= 10 acertos:
47.8%

>= 11:
21.4%

>= 12:
6.1%

>= 13:
0.8%

>= 14:
0%

15:
0%

F1:
0.XX

ROC-AUC:
0.XX

Brier Score:
0.XX
```

---

# 20. HISTÓRICO DOS TREINAMENTOS

Criar tabela persistente de experimentos.

Salvar:

```text
data do treinamento
versão
features utilizadas
modelo
hiperparâmetros
concursos de treino
concursos de teste
média de acertos
baseline
lift
F1
ROC-AUC
Brier Score
```

Isso permitirá saber se uma nova versão realmente melhorou.

---

# 21. MODEL VERSIONING

Cada treinamento deve possuir identificador.

Exemplo:

```text
LF-ENSEMBLE-v1.0
LF-ENSEMBLE-v1.1
LF-ENSEMBLE-v1.2
```

Nunca substituir silenciosamente o melhor modelo.

Guardar:

```text
modelo atual
melhor modelo
modelo anterior
```

---

# 22. PROMOÇÃO AUTOMÁTICA DO MODELO

Novo modelo só deve substituir o atual se:

```text
backtest >= modelo atual
```

e preferencialmente:

```text
resultado estatisticamente superior
```

Caso seja pior:

```text
manter modelo atual
```

Registrar o treinamento mesmo assim.

---

# 23. DETECTAR DATA LEAKAGE

Criar verificações automáticas.

Nenhuma feature utilizada para prever concurso N pode utilizar:

```text
concurso N
concurso N+1
qualquer concurso futuro
```

Adicionar testes automatizados para garantir isso.

---

# 24. REPRODUTIBILIDADE

Definir random seeds:

```python
random_state = 42
```

onde aplicável.

Salvar:

```text
versão dos dados
seed
hiperparâmetros
features
modelo
data
```

---

# 25. PERFORMANCE

Evitar recalcular todo histórico desnecessariamente.

Criar cache para features históricas.

Se possível:

```text
feature table
```

persistente por concurso.

Assim novos concursos devem recalcular apenas as partes necessárias.

---

# 26. ARQUITETURA FINAL

Desejo aproximadamente:

```text
BANCO DE CONCURSOS
        ↓
DATA VALIDATION
        ↓
FEATURE ENGINEERING
        ↓
TEMPORAL DATASET
        ↓
┌─────────────────────────────┐
│ Random Forest               │
│ Extra Trees                 │
│ Logistic Regression         │
│ Gradient Boosting           │
└─────────────────────────────┘
        ↓
CALIBRAÇÃO
        ↓
ENSEMBLE
        ↓
25 PROBABILIDADES
        ↓
RANKING
        ↓
TOP 15 / 16 / 17 / 18 / 19 / 20
        ↓
BACKTEST
        ↓
COMPARAÇÃO COM BASELINE
        ↓
VALIDAÇÃO ESTATÍSTICA
        ↓
MODELO APROVADO
```

---

# 27. PREPARAR PARA MÓDULO COMBINATÓRIO

O resultado do treinamento deve disponibilizar uma estrutura semelhante a:

```json
{
  "model_version": "LF-ENSEMBLE-v1.0",
  "ranking": [
    {
      "dezena": 7,
      "score": 0.714
    },
    {
      "dezena": 13,
      "score": 0.698
    }
  ],
  "top15": [],
  "top16": [],
  "top17": [],
  "top18": [],
  "top19": [],
  "top20": [],
  "metrics": {
    "mean_hits": 9.43,
    "baseline_hits": 9.00,
    "lift": 0.0478,
    "f1": 0.0,
    "roc_auc": 0.0,
    "brier": 0.0
  }
}
```

Posteriormente esse resultado será enviado para outro módulo responsável por:

```text
Monte Carlo
Distância de Hamming
Set Cover
Fechamentos
Otimização combinatória
```

---

# 28. TESTES

Criar testes automatizados para:

```text
feature engineering
ordem cronológica
ausência de leakage
geração das probabilidades
Top 15 contendo exatamente 15 dezenas
nenhuma dezena duplicada
valores entre 1 e 25
backtest
baseline
serialização do modelo
carregamento do modelo
```

---

# 29. NÃO QUEBRAR O SISTEMA EXISTENTE

Antes de alterar:

1. mapear dependências;
2. identificar endpoints utilizados pelo frontend;
3. preservar contratos existentes quando possível;
4. criar migration se houver mudança no banco;
5. criar fallback;
6. manter compatibilidade.

Não remover funções existentes sem verificar onde são utilizadas.

---

# 30. ENTREGA

Ao finalizar, apresente:

## Diagnóstico inicial

Explique por que o modelo anterior retornava aproximadamente:

```text
accuracy = 0.26
```

e diga exatamente como essa métrica estava sendo calculada.

## Mudanças realizadas

Liste arquivos alterados.

## Arquitetura

Explique o novo pipeline.

## Métricas

Apresente resultados do backtest.

## Comparação

```text
MODELO ANTIGO
vs
MODELO NOVO
vs
BASELINE ALEATÓRIO
```

## Segurança estatística

Informe explicitamente se existe ou não evidência de desempenho superior ao acaso.

---

# REGRA PRINCIPAL

Não quero que você apenas altere hiperparâmetros do RandomForest.

Quero uma revisão metodológica completa do sistema de treinamento.

Prioridades:

1. evitar data leakage;
2. validação temporal;
3. métricas corretas;
4. comparação com acaso;
5. probabilidades calibradas;
6. ensemble;
7. backtesting;
8. reprodutibilidade;
9. preparação para otimização combinatória.

Execute as alterações, teste o sistema e só considere concluído quando o novo pipeline estiver funcionando de ponta a ponta.
