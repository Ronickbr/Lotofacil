# PROMPT — SISTEMA DE GERAÇÃO DE JOGOS PARA LOTOFÁCIL

## OBJETIVO

Quero que você desenvolva ou melhore um módulo completo de geração de jogos para a Lotofácil, permitindo ao usuário criar apostas por diferentes estratégias matemáticas, estatísticas e combinatórias.

O sistema deve ser modular, permitindo combinar filtros, inteligência artificial, fechamentos, desdobramentos, cobertura e diversificação.

O objetivo NÃO é prometer previsão garantida de sorteios, mas criar jogos de forma estruturada, reproduzível e matematicamente organizada.

---

# 1. MODOS DE JOGO

Implemente os seguintes modos principais:

## 1.1 JOGO DIRETO

Permitir ao usuário selecionar manualmente exatamente 15 dezenas entre 1 e 25.

Regras:

- exatamente 15 dezenas;
- nenhuma repetida;
- validar intervalo de 1 a 25;
- ordenar as dezenas;
- salvar o jogo;
- permitir gerar novamente;
- permitir adicionar o jogo a uma lista de apostas.

Exemplo:

```text
01 02 03 05 06
08 09 10 12 13
15 18 20 22 25
```

---

## 1.2 JOGO AMPLIADO

Permitir selecionar:

- 16 dezenas;
- 17 dezenas;
- 18 dezenas;
- 19 dezenas;
- 20 dezenas.

O sistema deve informar quantas combinações de 15 dezenas existem dentro da seleção.

Usar:

\[
C(n,15)=\frac{n!}{15!(n-15)!}
\]

Exemplos:

```text
16 dezenas = 16 combinações
17 dezenas = 136 combinações
18 dezenas = 816 combinações
19 dezenas = 3.876 combinações
20 dezenas = 15.504 combinações
```

Exibir claramente:

- quantidade de dezenas escolhidas;
- total de combinações possíveis;
- total de jogos que serão gerados;
- custo estimado, se o valor unitário da aposta estiver cadastrado no sistema.

---

# 2. JOGO BALANCEADO

Criar um modo que gere jogos respeitando filtros configuráveis.

Permitir configurar:

## Pares e ímpares

Exemplos:

```text
7 pares / 8 ímpares
8 pares / 7 ímpares
6 pares / 9 ímpares
9 pares / 6 ímpares
```

Permitir múltiplas opções simultaneamente.

---

## Soma das dezenas

Permitir definir intervalo:

```text
Soma mínima
Soma máxima
```

Exemplo:

```text
180 até 220
```

---

## Linhas

Considerar o volante 5x5:

```text
01 02 03 04 05
06 07 08 09 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

Permitir controlar a quantidade de dezenas por linha.

---

## Colunas

Permitir controlar a quantidade por coluna.

---

## Sequências consecutivas

Permitir configurar:

```text
quantidade mínima de consecutivos
quantidade máxima
maior sequência permitida
```

Exemplo:

```text
máximo de 4 consecutivos
```

---

## Repetição do último concurso

Permitir:

```text
mínimo de dezenas repetidas
máximo de dezenas repetidas
```

Exemplo:

```text
8 a 10 dezenas repetidas
```

---

## Faixas numéricas

Separar:

```text
01–05
06–10
11–15
16–20
21–25
```

Permitir definir quantidade mínima e máxima em cada faixa.

---

# 3. JOGO POR FREQUÊNCIA

Criar modo baseado em frequência histórica.

Calcular para cada dezena:

```text
frequência últimos 5 concursos
frequência últimos 10
frequência últimos 20
frequência últimos 50
frequência últimos 100
frequência histórica
```

Classificar dezenas como:

```text
quentes
neutras
frias
```

Permitir ao usuário configurar composição.

Exemplo:

```text
5 quentes
6 neutras
4 frias
```

Não afirmar que dezenas mais frequentes possuem necessariamente maior chance real no próximo sorteio.

---

# 4. JOGO POR ATRASO

Calcular:

```text
atraso atual
atraso médio
atraso máximo
percentil do atraso
```

Permitir gerar jogos utilizando:

```text
dezenas mais atrasadas
dezenas menos atrasadas
mistura balanceada
```

Permitir definir pesos.

---

# 5. JOGO MISTO

Criar modo que combine critérios.

Exemplo:

```text
30% frequência
20% atraso
20% tendência
15% repetição
15% associação histórica
```

Gerar um score para cada dezena:

\[
Score(d)=
w_1F+
w_2A+
w_3T+
w_4R+
w_5C
\]

onde:

```text
F = frequência
A = atraso
T = tendência
R = repetição
C = correlação/associação
```

Normalizar os componentes antes da soma.

---

# 6. JOGO POR REPETIÇÃO

Utilizar o concurso anterior.

Calcular:

\[
R=|J_t \cap J_{t-1}|
\]

Permitir escolher:

```text
7 repetidas
8 repetidas
9 repetidas
10 repetidas
11 repetidas
```

ou intervalo.

Exemplo:

```text
entre 8 e 10 repetidas
```

---

# 7. JOGO POR LINHAS E COLUNAS

Criar gerador específico baseado na matriz 5x5.

Permitir:

```text
mínimo por linha
máximo por linha
mínimo por coluna
máximo por coluna
```

Também permitir evitar concentração excessiva em uma região do volante.

---

# 8. JOGO POR PADRÕES

Criar filtros opcionais para:

```text
números primos
Fibonacci
múltiplos de 3
múltiplos de 5
quadrados perfeitos
borda
centro
diagonais
sequências
```

Permitir definir mínimo e máximo de cada característica.

Esses padrões devem ser tratados como filtros descritivos e não como vantagem probabilística garantida.

---

# 9. JOGO POR IA

Integrar com o módulo de inteligência artificial existente.

A IA deve retornar uma probabilidade ou score para cada uma das 25 dezenas.

Exemplo:

```text
01 -> 0.721
02 -> 0.603
03 -> 0.692
...
25 -> 0.688
```

Ordenar:

```text
Top 15
Top 16
Top 17
Top 18
Top 19
Top 20
```

Permitir gerar jogos usando:

```text
Top 15 puro
Top 18 com desdobramento
Top 20 com fechamento
Top N com filtros adicionais
```

---

# 10. JOGO DIVERSIFICADO

Quando o usuário solicitar vários jogos, evitar que sejam quase idênticos.

Usar distância de Hamming.

Representar cada jogo como vetor binário de 25 posições.

Exemplo:

```text
101101001...
```

Calcular:

\[
d_H(A,B)
\]

O objetivo deve ser maximizar:

\[
\max \min_{i\neq j} d_H(J_i,J_j)
\]

Permitir configurar:

```text
diversidade baixa
diversidade média
diversidade alta
```

Quanto maior a diversidade, menor a sobreposição média entre os jogos.

---

# 11. JOGO POR COBERTURA MÁXIMA

Criar modo baseado em cobertura combinatória.

O usuário informa:

```text
quantidade de jogos
grupo de dezenas
objetivo de cobertura
```

Exemplo:

```text
18 dezenas candidatas
30 jogos
maximizar cobertura de combinações de 12 dezenas
```

Utilizar técnicas como:

```text
Set Cover
Greedy Set Cover
Integer Programming
algoritmo genético
simulated annealing
```

Escolher a técnica mais eficiente de acordo com o tamanho do problema.

---

# 12. DESDOBRAMENTO

Criar módulo dedicado.

Exemplo:

```text
18 dezenas
↓
gerar combinações de 15
```

Permitir:

```text
desdobramento completo
desdobramento reduzido
desdobramento otimizado
```

No completo:

\[
C(18,15)=816
\]

No reduzido, gerar apenas subconjunto estratégico.

---

# 13. FECHAMENTO MATEMÁTICO

Criar modo de fechamento.

Permitir configurar:

```text
quantidade de dezenas base
quantidade de dezenas por jogo
objetivo de premiação
número máximo de jogos
```

Exemplo:

```text
18 dezenas base
15 por jogo
objetivo: maximizar cobertura para 13 acertos
máximo: 50 jogos
```

O sistema deve exibir claramente que garantias de fechamento são condicionais.

Exemplo:

```text
Garantia válida somente se X dezenas sorteadas estiverem dentro do conjunto-base.
```

Nunca apresentar garantia absoluta.

---

# 14. FECHAMENTO REDUZIDO

Criar algoritmo que encontre subconjunto dos jogos completos.

Objetivo:

```text
reduzir custo
mantendo máxima cobertura possível
```

Exemplo:

```text
Universo:
816 combinações

Jogos selecionados:
40

Cobertura:
X%
```

---

# 15. MONTE CARLO

Implementar simulador Monte Carlo para avaliar conjuntos de jogos.

Permitir executar:

```text
100.000 simulações
500.000 simulações
1.000.000 simulações
```

Para cada conjunto de apostas, medir:

```text
média de acertos
mediana
melhor resultado
probabilidade empírica de >= 11
>= 12
>= 13
>= 14
15
```

Comparar diferentes estratégias.

---

# 16. ALGORITMO GENÉTICO

Criar algoritmo genético opcional.

Cada jogo será um cromossomo de 25 bits.

Restrição:

\[
\sum x_i = 15
\]

Fitness pode considerar:

```text
score da IA
diversidade
cobertura
filtros estatísticos
penalidade por sobreposição
```

Exemplo:

\[
Fitness =
w_1ScoreIA+
w_2Diversidade+
w_3Cobertura-
w_4Sobreposicao
\]

---

# 17. MODOS DISPONÍVEIS NA INTERFACE

Criar os seguintes cartões ou opções:

```text
Jogo Direto
Jogo Ampliado
Jogo Balanceado
Jogo por Frequência
Jogo por Atraso
Jogo Misto
Jogo por Repetição
Jogo por Padrões
Jogo por IA
Jogo Diversificado
Jogo por Cobertura Máxima
Desdobramento
Fechamento Matemático
Fechamento Reduzido
```

Cada modo deve possuir:

```text
nome
descrição
nível de complexidade
configurações
quantidade de jogos
resultado
```

---

# 18. GERADOR MULTIESTRATÉGIA

Criar opção avançada:

```text
Gerador Inteligente
```

O usuário pode combinar várias estratégias.

Exemplo:

```text
IA = ON
Balanceamento = ON
Repetição = 8–10
Soma = 180–220
Diversidade = Alta
Cobertura = Máxima
```

Pipeline:

```text
HISTÓRICO
↓
ANÁLISE ESTATÍSTICA
↓
SCORE DAS DEZENAS
↓
FILTROS
↓
GERAÇÃO DE CANDIDATOS
↓
DIVERSIFICAÇÃO
↓
COBERTURA
↓
MONTE CARLO
↓
RANKING FINAL
```

---

# 19. RANKING DOS JOGOS

Cada jogo gerado deve receber score.

Exemplo:

```text
JOGO 01

01 03 04 05 07
08 10 11 13 14
16 18 21 23 25

Score IA: 87.4
Balanceamento: 92
Diversidade: 89
Cobertura: 94

Score final: 90.6
```

Permitir ordenar por:

```text
score geral
IA
diversidade
cobertura
balanceamento
```

---

# 20. EXPLICAÇÃO DO JOGO

Para cada jogo, gerar uma explicação curta.

Exemplo:

```text
7 pares / 8 ímpares
9 repetidas do último concurso
soma: 197
3 primos
distribuição equilibrada entre linhas
distância média dos demais jogos: 8
```

---

# 21. COMPARADOR DE ESTRATÉGIAS

Criar tela comparativa.

Exemplo:

```text
                    IA     BALANCEADO     COBERTURA

Jogos               20         20             20
Média Monte Carlo   9.3        9.1            9.2
Diversidade         72%        81%            94%
Cobertura           68%        75%            96%
```

---

# 22. NÃO DUPLICAR JOGOS

Criar hash único para cada combinação.

Antes de salvar:

```text
ordenar dezenas
gerar hash
verificar duplicidade
```

Nenhum jogo idêntico pode ser criado duas vezes dentro da mesma geração.

---

# 23. VALIDAÇÃO

Todo jogo deve respeitar:

```text
15 dezenas
valores de 1 a 25
nenhuma repetida
ordem crescente
```

Para jogos ampliados, respeitar o tamanho definido.

---

# 24. PERSISTÊNCIA

Salvar:

```text
id
data
estratégia
configuração
dezenas
score
modelo IA utilizado
versão do algoritmo
resultado do Monte Carlo
```

---

# 25. HISTÓRICO DE RESULTADOS

Quando novos concursos forem cadastrados, comparar os jogos antigos.

Salvar:

```text
quantidade de acertos
concurso
premiação, se cadastrada
estratégia utilizada
```

Isso permitirá comparar estratégias ao longo do tempo.

---

# 26. RELATÓRIO DE DESEMPENHO

Criar relatório:

```text
Estratégia
Jogos gerados
Concursos avaliados
Média de acertos
Maior acerto
Quantidade de 11
Quantidade de 12
Quantidade de 13
Quantidade de 14
Quantidade de 15
```

---

# 27. BASELINE

Sempre comparar algoritmos avançados com jogos aleatórios.

Criar baseline:

```text
15 dezenas escolhidas aleatoriamente
```

Executar simulações equivalentes.

Exibir:

```text
Estratégia vs Aleatório
```

---

# 28. REGRA DE TRANSPARÊNCIA

Nunca exibir frases como:

```text
"Jogo garantido"
"Estas dezenas certamente sairão"
"Probabilidade garantida de prêmio"
```

Utilizar:

```text
"Jogo gerado com base nos critérios selecionados."
```

ou:

```text
"Otimização combinatória não altera a aleatoriedade do sorteio."
```

---

# 29. PERFORMANCE

Para grandes quantidades de combinações:

- evitar carregar todas em memória quando não for necessário;
- utilizar generators/iterators;
- processamento em lote;
- cache;
- paralelização quando apropriado.

---

# 30. TESTES AUTOMATIZADOS

Criar testes para:

```text
combinações
desdobramentos
filtros
distância de Hamming
set cover
Monte Carlo
scores
validação
duplicidade
persistência
```

---

# 31. ARQUITETURA RECOMENDADA

```text
BANCO DE CONCURSOS
        ↓
ANÁLISE ESTATÍSTICA
        ↓
MODELO IA
        ↓
RANKING DAS 25 DEZENAS
        ↓
GERADOR DE CANDIDATOS
        ↓
FILTROS
        ↓
DESDOBRAMENTO / FECHAMENTO
        ↓
DIVERSIFICAÇÃO
        ↓
COBERTURA
        ↓
MONTE CARLO
        ↓
RANKING DOS JOGOS
        ↓
RESULTADO FINAL
```

---

# 32. ENTREGA

Ao finalizar:

1. analise o sistema atual;
2. identifique o que já existe;
3. preserve funcionalidades compatíveis;
4. implemente os novos modos;
5. crie testes;
6. execute os testes;
7. documente arquivos alterados;
8. apresente exemplos de geração;
9. compare estratégias;
10. confirme que o sistema funciona ponta a ponta.

Não faça apenas uma interface visual.

Implemente também a lógica matemática por trás de cada estratégia.

---

# PRIORIDADES

Prioridade máxima:

```text
1. Correção matemática
2. Validação dos jogos
3. Diversificação
4. Cobertura combinatória
5. Integração com IA
6. Monte Carlo
7. Persistência
8. Comparação de estratégias
9. Performance
10. Interface
```

O módulo final deve permitir desde uma aposta simples de 15 dezenas até estratégias avançadas com IA, fechamentos, desdobramentos, cobertura máxima e diversificação.
