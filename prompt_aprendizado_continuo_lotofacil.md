# MÓDULO DE APRENDIZADO CONTÍNUO — LOTOFÁCIL

## OBJETIVO

Quero implementar no sistema um mecanismo completo de:

GERAÇÃO  
→ REGISTRO DOS JOGOS  
→ AGUARDAR PRÓXIMO CONCURSO  
→ IMPORTAR RESULTADO OFICIAL  
→ CONFERIR JOGOS  
→ MEDIR DESEMPENHO  
→ ATUALIZAR DATASET  
→ RETREINAR  
→ COMPARAR MODELOS  
→ PROMOVER SOMENTE SE HOUVER MELHORIA  
→ GERAR NOVA GERAÇÃO DE JOGOS

O sistema deverá criar um ciclo permanente de aprendizado e avaliação.

IMPORTANTE:

Não quero que o sistema simplesmente modifique o modelo com base em um único concurso.

O aprendizado deverá considerar desempenho acumulado, backtesting temporal, validação estatística e comparação contra baseline aleatório.

---

## 1. REGISTRAR TODA GERAÇÃO

Sempre que forem gerados jogos, criar uma sessão de geração.

Exemplo:

```text
Geração:
GEN-2026-001245

Concurso alvo:
XXXX

Data:
YYYY-MM-DD HH:MM

Modelo:
LF-ENSEMBLE-v1.4

Estratégia:
IA + Cobertura + Diversificação

Quantidade:
30 jogos
```

Salvar permanentemente:

- ID da geração;
- concurso alvo;
- data/hora;
- jogos;
- estratégia;
- filtros;
- parâmetros;
- versão do modelo;
- scores;
- ranking das dezenas;
- probabilidades das 25 dezenas;
- configuração combinatória;
- seed utilizada;
- versão do algoritmo;
- status da geração.

Status possíveis:

```text
AGUARDANDO_RESULTADO
RESULTADO_DISPONIVEL
CONFERIDO
AVALIADO
UTILIZADO_NO_TREINAMENTO
```

---

## 2. NÃO ALTERAR JOGOS ANTIGOS

Após a geração de um jogo para determinado concurso:

BLOQUEAR alterações.

O jogo precisa permanecer exatamente como foi criado antes do sorteio.

Isso é obrigatório para garantir auditoria e evitar:

DATA LEAKAGE

ou alteração retroativa de previsões.

Adicionar:

```text
created_at
locked_at
hash
```

Gerar hash para cada jogo e geração.

---

## 3. CONCURSO ALVO

Todo jogo deve obrigatoriamente possuir:

```text
target_contest
```

Exemplo:

```json
{
  "generation_id": "GEN-4521",
  "target_contest": 4001,
  "model_version": "LF-ENSEMBLE-v1.4"
}
```

Nunca avaliar um jogo contra concurso diferente daquele para o qual foi criado.

---

## 4. BUSCAR RESULTADO DO PRÓXIMO CONCURSO

Criar serviço:

```text
LotteryResultService
```

Responsável por verificar se o resultado oficial do concurso alvo está disponível.

Fluxo:

```text
GERAÇÃO PARA CONCURSO 4001
↓
AGUARDANDO
↓
RESULTADO 4001 DISPONÍVEL
↓
IMPORTAR RESULTADO
↓
VALIDAR
↓
SALVAR
↓
CONFERIR JOGOS
```

Utilizar fonte oficial ou fonte confiável configurada no sistema.

Nunca sobrescrever silenciosamente um resultado já armazenado.

---

## 5. VALIDAR RESULTADO

Antes de processar, confirmar:

```text
15 dezenas
sem duplicação
números entre 1 e 25
concurso correto
data correta
```

Ordenar dezenas.

Exemplo:

```text
01 02 04 05 07
08 10 12 13 15
16 18 21 23 25
```

---

## 6. CONFERÊNCIA AUTOMÁTICA

Para cada jogo:

\[
Hits = |Jogo \cap Resultado|
\]

Registrar:

```text
jogo
concurso
acertos
dezenas acertadas
dezenas erradas
dezenas sorteadas não escolhidas
```

Exemplo:

```text
JOGO 018

Previstas:
01 02 03 05 07 08 10 11 13 14 16 18 20 23 25

Resultado:
01 02 04 05 07 08 10 12 13 14 16 18 21 23 25

Acertos:
12 / 15

Acertadas:
01 02 05 07 08 10 13 14 16 18 23 25

Não acertadas:
03 11 20

Faltaram:
04 12 21
```

---

## 7. CLASSIFICAR RESULTADOS

Registrar automaticamente:

```text
11 acertos
12 acertos
13 acertos
14 acertos
15 acertos
```

Também guardar resultados abaixo de 11 para fins estatísticos.

---

## 8. AVALIAR A GERAÇÃO COMPLETA

Não analisar apenas o melhor jogo.

Avaliar toda a geração.

Calcular:

```text
quantidade de jogos
média de acertos
mediana
desvio padrão
mínimo
máximo
```

Distribuição:

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

E:

```text
% >= 10
% >= 11
% >= 12
% >= 13
% >= 14
% = 15
```

---

## 9. AVALIAR RANKING DA IA

Guardar o ranking das 25 dezenas produzido ANTES do concurso.

Depois verificar:

```text
quantas sorteadas estavam no Top 15
Top 16
Top 17
Top 18
Top 19
Top 20
```

Exemplo:

```text
Top 15 → 10/15
Top 16 → 11/15
Top 17 → 12/15
Top 18 → 13/15
Top 19 → 14/15
Top 20 → 14/15
```

Isso será uma das principais métricas para avaliar o modelo preditivo.

---

## 10. CALIBRAÇÃO DAS PROBABILIDADES

Guardar:

```text
probabilidade prevista
resultado real
```

Exemplo:

```text
07 → 0.72 → saiu
13 → 0.69 → saiu
24 → 0.67 → não saiu
```

Com os concursos acumulados calcular:

```text
Brier Score
Log Loss
Calibration Error
Calibration Curve
```

Assim poderemos descobrir se o modelo está excessivamente confiante.

---

## 11. CRIAR DATASET DE PREVISÕES

Criar tabela:

```text
prediction_history
```

Estrutura:

```text
contest
number
predicted_probability
ranking_position
actual_result
model_version
generation_id
created_at
```

Exemplo:

```text
4001 | 01 | 0.684 | 05 | 1 | v1.4
4001 | 02 | 0.621 | 11 | 1 | v1.4
4001 | 03 | 0.598 | 15 | 0 | v1.4
```

Esse dataset será essencial para avaliar o desempenho real fora da amostra.

---

## 12. HISTÓRICO DE PERFORMANCE

Criar tabela:

```text
model_performance
```

Registrar:

```text
model_version
contest
top15_hits
top16_hits
top17_hits
top18_hits
top19_hits
top20_hits
mean_game_hits
best_game_hits
f1
roc_auc
brier
log_loss
baseline_difference
created_at
```

---

## 13. DASHBOARD DE EVOLUÇÃO

Criar painel:

```text
EVOLUÇÃO DO MODELO
```

Exemplo:

```text
Modelo atual:
LF-ENSEMBLE-v1.6

Concursos avaliados:
72

Top15 médio:
9.42

Baseline:
9.00

Lift:
+4.67%

Melhor Top15:
13

Brier Score:
0.231
```

Mostrar gráfico temporal:

```text
Acertos Top15 por concurso
```

E média móvel:

```text
últimos 10
últimos 20
últimos 50
histórico
```

---

## 14. NÃO APRENDER COM UM ÚNICO CONCURSO

REGRA CRÍTICA.

Um resultado isolado NÃO pode causar alteração estrutural no modelo.

Utilizar janela mínima configurável.

Exemplo:

```text
mínimo 20 novos concursos
```

antes de realizar uma avaliação completa de retreinamento.

Permitir configurar:

```text
20
30
50
100
```

concursos.

---

## 15. ATUALIZAÇÃO INCREMENTAL DO DATASET

Quando um novo concurso ocorrer:

```text
resultado oficial
↓
salvar
↓
conferir
↓
adicionar ao histórico
↓
recalcular features necessárias
```

Não reconstruir todo o dataset se não for necessário.

---

## 16. RETREINAMENTO AUTOMÁTICO

Criar:

```text
ContinuousTrainingService
```

Pode iniciar quando:

```text
N novos concursos
```

forem acumulados.

Exemplo:

```text
20 novos concursos
→ criar candidato
```

NÃO substituir imediatamente o modelo atual.

---

## 17. CHAMPION / CHALLENGER

Utilizar arquitetura:

```text
CHAMPION
modelo atualmente utilizado

CHALLENGER
novo modelo treinado
```

Exemplo:

```text
Champion:
LF-ENSEMBLE-v1.6

Challenger:
LF-ENSEMBLE-v1.7
```

Comparar os dois.

---

## 18. BACKTEST DO CHALLENGER

Antes de promover:

executar Walk-Forward Backtest.

Comparar:

```text
Champion
Challenger
Baseline aleatório
```

Em exatamente os mesmos concursos.

---

## 19. CRITÉRIOS DE PROMOÇÃO

O Challenger só pode substituir o Champion se houver melhoria consistente.

Considerar:

```text
Top15 médio
Top18 coverage
Brier Score
Log Loss
estabilidade
performance recente
performance histórica
```

Exemplo:

```text
Champion
Top15 = 9.31

Challenger
Top15 = 9.47
```

Mas não promover apenas pela diferença absoluta.

Realizar testes estatísticos.

---

## 20. SIGNIFICÂNCIA ESTATÍSTICA

Implementar:

```text
bootstrap
permutation test
confidence interval
```

Calcular:

```text
IC 95%
```

Exemplo:

```text
Diferença Challenger - Champion:

+0.16 dezenas

IC 95%:
+0.04 até +0.28
```

Somente considerar melhoria real quando houver evidência suficiente.

---

## 21. BASELINE ALEATÓRIO

Sempre comparar também com baseline.

Para Top15:

\[
E[X]=15 \times \frac{15}{25}=9
\]

Executar também simulação Monte Carlo.

Exemplo:

```text
Aleatório:
9.00

Champion:
9.31

Challenger:
9.47
```

---

## 22. SISTEMA DE PONTUAÇÃO DO MODELO

Criar:

```text
MODEL_SCORE
```

Exemplo conceitual:

\[
Score =
w_1 Hits15 +
w_2 Hits18 +
w_3 Calibration +
w_4 Stability +
w_5 Lift
\]

Normalizar métricas antes da combinação.

Não usar apenas uma métrica.

---

## 23. PESOS AUTOAJUSTÁVEIS

Caso exista Ensemble:

```text
Random Forest
Extra Trees
Logistic
Gradient Boosting
XGBoost
LightGBM
```

registrar desempenho individual de cada modelo.

Permitir ajustar pesos com base no desempenho histórico.

Exemplo:

```text
Random Forest       0.20
Extra Trees         0.18
Logistic            0.27
Gradient Boosting   0.21
XGBoost             0.14
```

Mas esses pesos devem ser calculados exclusivamente usando dados anteriores ao período avaliado.

---

## 24. PERFORMANCE POR JANELA

Analisar:

```text
últimos 10 concursos
últimos 20
últimos 50
últimos 100
histórico total
```

Não permitir que desempenho muito recente domine todo o treinamento.

---

## 25. DETECÇÃO DE OVERFITTING

Se Challenger apresentar:

```text
excelente treino
excelente backtest interno
péssimo desempenho recente real
```

marcar:

```text
POSSIBLE_OVERFITTING
```

Não promover.

---

## 26. DETECÇÃO DE REGRESSÃO

Caso modelo novo seja pior:

```text
MODEL_REGRESSION
```

Manter Champion.

Registrar Challenger para auditoria.

---

## 27. ROLLBACK

Manter sempre:

```text
Champion atual
Champion anterior
Challengers
```

Se uma nova versão apresentar regressão grave após promoção:

permitir rollback.

Exemplo:

```text
v1.6
↓
v1.7 promovido
↓
regressão detectada
↓
rollback v1.6
```

---

## 28. VERSIONAMENTO

Utilizar:

```text
LF-ENSEMBLE-v1.0
LF-ENSEMBLE-v1.1
LF-ENSEMBLE-v1.2
```

Salvar para cada versão:

```text
dataset
features
hiperparâmetros
modelos
pesos
seed
data treinamento
métricas
git commit, se disponível
```

---

## 29. AUDITORIA COMPLETA

Nunca apagar resultados ruins.

Precisamos saber exatamente:

```text
o que foi previsto
quando
por qual modelo
com quais parâmetros
qual foi o resultado
```

Isso evita viés de sobrevivência.

---

## 30. APRENDIZADO DAS ESTRATÉGIAS DE JOGO

Além da IA das dezenas, avaliar cada estratégia combinatória.

Exemplo:

```text
IA puro
IA + balanceamento
IA + Hamming
IA + Set Cover
IA + Monte Carlo
Fechamento
Frequência
Misto
```

Criar ranking histórico.

---

## 31. SCORE DAS ESTRATÉGIAS

Exemplo:

```text
Estratégia               Média

IA + Cobertura             9.62
IA + Diversidade           9.48
IA puro                    9.31
Balanceado                 9.17
Aleatório                  9.00
```

Não concluir superioridade apenas pela média.

Mostrar:

```text
amostra
intervalo de confiança
variância
```

---

## 32. MULTI-ARMED BANDIT OPCIONAL

Depois que houver histórico suficiente, estudar utilização de:

```text
Multi-Armed Bandit
```

para distribuir parte dos jogos entre estratégias.

Exemplo:

```text
50 jogos

20 → estratégia A
15 → estratégia B
10 → estratégia C
5 → exploração
```

Manter sempre componente de exploração.

Não deixar uma estratégia dominar apenas por resultados recentes.

---

## 33. EXPLORATION VS EXPLOITATION

Implementar conceito:

```text
EXPLOITATION
usar estratégias historicamente melhores

EXPLORATION
continuar testando alternativas
```

Exemplo:

```text
80% exploitation
20% exploration
```

Configurável.

---

## 34. REGISTRAR EXPERIMENTOS

Criar:

```text
experiment_history
```

Exemplo:

```text
EXP-204

Hipótese:
Adicionar peso maior para Logistic Regression.

Champion:
v1.6

Challenger:
v1.7

Resultado:
+0.08 Top15

Significância:
não significativa

Decisão:
REJEITADO
```

---

## 35. CICLO DE APRENDIZADO

Implementar fluxo:

```text
┌──────────────────────┐
│ RESULTADO OFICIAL    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ CONFERIR JOGOS       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ CALCULAR MÉTRICAS    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ ATUALIZAR DATASET    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ ATUALIZAR FEATURES   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ TREINAR CHALLENGER   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ WALK-FORWARD TEST    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ VS CHAMPION          │
│ VS RANDOM            │
└──────────┬───────────┘
           ↓
       MELHOROU?
        ↙     ↘
      NÃO     SIM
       ↓       ↓
 REJEITAR   PROMOVER
               ↓
       NOVO CHAMPION
               ↓
       GERAR PRÓXIMOS
            JOGOS
```

---

## 36. AUTOMAÇÃO DO PRÓXIMO CONCURSO

Após processar concurso N:

automaticamente preparar:

```text
target_contest = N + 1
```

Executar:

```text
atualizar histórico
gerar features
carregar Champion
gerar probabilidades
criar ranking
gerar jogos
salvar geração
bloquear geração
aguardar resultado
```

---

## 37. DASHBOARD DO CICLO

Criar seção:

```text
APRENDIZADO CONTÍNUO
```

Mostrar:

```text
Último concurso processado
Próximo concurso alvo
Modelo Champion
Último Challenger
Resultado da comparação
Quantidade de concursos aprendidos
Último treinamento
Próximo treinamento previsto
```

---

## 38. PAINEL DO ÚLTIMO CONCURSO

Exemplo:

```text
CONCURSO 4001

Jogos avaliados:
50

Melhor jogo:
13 acertos

Média:
9.48

>= 11:
12 jogos

>= 12:
4 jogos

>= 13:
1 jogo

Top15 IA:
10/15

Top18 IA:
13/15
```

---

## 39. EVOLUÇÃO DO MODELO

Criar tabela:

| Versão | Top15 | Top18 | Brier | Lift | Status |
|---|---:|---:|---:|---:|---|
| v1.4 | 9.18 | 10.94 | 0.241 | +2.0% | Arquivado |
| v1.5 | 9.29 | 11.07 | 0.236 | +3.2% | Arquivado |
| v1.6 | 9.41 | 11.21 | 0.229 | +4.6% | Champion |
| v1.7 | 9.44 | 11.19 | 0.232 | +4.9% | Challenger |

---

## 40. ALERTAS

Criar alertas:

```text
NOVO_RESULTADO
CONFERENCIA_CONCLUIDA
NOVO_RECORDE
CHALLENGER_CRIADO
CHALLENGER_PROMOVIDO
CHALLENGER_REJEITADO
MODEL_REGRESSION
DATA_ERROR
RESULT_SOURCE_ERROR
```

---

## 41. REGRA FUNDAMENTAL CONTRA DATA LEAKAGE

Ao prever concurso N:

NENHUMA informação do concurso N ou posterior pode existir nas features.

Formalmente:

\[
Feature(N)=f(1,\ldots,N-1)
\]

Nunca:

\[
Feature(N)=f(1,\ldots,N)
\]

Criar testes automatizados para garantir isso.

---

## 42. REGRA FUNDAMENTAL DE AUTOEVOLUÇÃO

O sistema deve tentar melhorar continuamente, mas:

NÃO modificar automaticamente produção apenas porque o último concurso teve resultado ruim.

NÃO promover modelo apenas porque acertou mais em poucos concursos.

NÃO otimizar diretamente para resultados isolados.

Somente promover alterações que apresentem:

```text
melhoria fora da amostra
+
estabilidade
+
ausência de leakage
+
backtest temporal
+
comparação com Champion
+
comparação com baseline
```

---

## 43. META-APRENDIZADO

Depois de acumular histórico suficiente, permitir que o sistema descubra quais features são realmente úteis.

Registrar:

```text
feature importance
permutation importance
SHAP, se disponível
```

Comparar importância ao longo das versões.

Excluir gradualmente features que:

```text
não agregam desempenho
aumentam overfitting
são redundantes
```

---

## 44. ABLATION TESTING

Antes de adicionar permanentemente uma nova feature, testar:

```text
modelo sem feature
VS
modelo com feature
```

Exemplo:

```text
Baseline:
9.37

+ atraso:
9.39

+ pares/trincas:
9.38

+ média móvel:
9.44
```

Assim identificar quais componentes realmente contribuem.

---

## 45. AUTOEXPERIMENTAÇÃO CONTROLADA

Permitir que o sistema crie Challengers automaticamente testando:

```text
novas janelas
novos pesos
novos hiperparâmetros
novas combinações de modelos
remoção de features
novas features
```

Limitar número de experimentos por ciclo.

Todos os experimentos devem ser reproduzíveis.

---

## 46. NÃO CONFUNDIR SORTE COM APRENDIZADO

O sistema deverá diferenciar:

```text
resultado melhor por acaso
```

de:

```text
melhoria persistente do modelo
```

Utilizar tamanho de amostra, intervalos de confiança e backtesting.

---

## 47. RESULTADO FINAL DO CICLO

Após cada concurso produzir relatório:

```json
{
  "contest": 4001,
  "generation": "GEN-4001-A",
  "model": "LF-ENSEMBLE-v1.6",
  "games": 50,
  "mean_hits": 9.48,
  "best_hits": 13,
  "top15_hits": 10,
  "top18_hits": 13,
  "baseline_mean": 9.0,
  "model_status": "CHAMPION",
  "retraining_required": false
}
```

---

## 48. OBJETIVO FINAL

Quero transformar o sistema em um ciclo permanente:

```text
PREVER
↓
REGISTRAR
↓
AGUARDAR
↓
CONFERIR
↓
MEDIR
↓
APRENDER
↓
TESTAR
↓
VALIDAR
↓
MELHORAR
↓
PREVER NOVAMENTE
```

O histórico de previsões reais deverá ser considerado mais importante do que métricas obtidas apenas durante treinamento.

O sistema deve buscar melhoria contínua sem sacrificar rigor estatístico, auditabilidade e proteção contra overfitting.

---

## 49. SEPARAR DESEMPENHO DO MODELO E DO GERADOR

Criar dois placares independentes:

```text
DESEMPENHO DO MODELO
---------------------
Top15
Top16
Top17
Top18
Calibração
Brier Score
ROC-AUC
```

e:

```text
DESEMPENHO DO GERADOR
----------------------
Média dos jogos
Melhor jogo
11 acertos
12 acertos
13 acertos
14 acertos
15 acertos
Diversidade
Cobertura
```

Isso permitirá distinguir:

- falha na previsão das dezenas;
- falha na montagem combinatória dos jogos.

Exemplo:

```text
IA colocou 13 das 15 sorteadas no Top18
mas o melhor jogo gerado teve apenas 11 acertos
```

Nesse cenário, não retreinar automaticamente a IA.

Primeiro avaliar e melhorar o gerador combinatório.

---

## 50. META-OTIMIZADOR

Criar uma terceira camada chamada:

```text
MetaOptimizer
```

Responsável por comparar combinações de:

```text
modelo preditivo
+
estratégia de geração
+
filtros
+
método de cobertura
+
nível de diversidade
```

Exemplo:

```text
LF-ENSEMBLE-v1.6
+
Set Cover
+
Hamming alto
+
Filtro de repetição
```

contra:

```text
LF-ENSEMBLE-v1.6
+
Algoritmo Genético
+
Monte Carlo
+
Filtro balanceado
```

Registrar desempenho histórico de cada combinação.

---

# REGRA PRINCIPAL

Não quero apenas um sistema que confira apostas.

Quero um sistema auditável de aprendizado contínuo que:

1. registre cada previsão antes do sorteio;
2. bloqueie alterações posteriores;
3. confira automaticamente;
4. meça o desempenho real;
5. compare com baseline;
6. retreine somente com histórico suficiente;
7. crie Challengers;
8. faça backtesting temporal;
9. promova apenas versões melhores;
10. permita rollback;
11. compare estratégias combinatórias;
12. aprenda sem perseguir ruído aleatório.

Execute a implementação ponta a ponta e crie testes automatizados para todo o fluxo.
