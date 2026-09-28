# 🍀 Sistema de Análise da Lotofácil — Analytics Pro

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-5.3-purple)

Uma plataforma avançada para análise estatística, gestão de resultados, aprendizado contínuo (MLOps) e geração inteligente de cartões para a Lotofácil utilizando Machine Learning e combinações matemáticas.

---

## 📖 Sobre o Projeto

Este sistema foi desenvolvido para oferecer ferramentas completas de análise estatística e geração estratégica dos resultados da Lotofácil. Ele combina análises probabilísticas avançadas (estatística bayesiana, filtros de Monte Carlo e desdobramentos combinatórios) com modelos de aprendizado de máquina (*RandomForest* e algoritmos genéticos).

---

## ✨ Funcionalidades Principais

- **🏭 Fábrica de Jogos (`/smart-generate`):** Interface redesenhada com Hero Banner, menu lateral de estratégias (*Jogo Direto*, *Estatística*, *Desdobramento Combinatorial*, *IA* e *Monte Carlo*), volante numérico 3D interativo, visualização dos cartões em grid de 2 colunas com esferas codificadas por cor e barra flutuante para salvamento no banco.
- **💾 Meus Jogos (`/saved-games`):** Gerenciamento e acompanhamento dos cartões salvos exibidos em layout responsivo de 2 colunas, com comparativo de acertos em relação ao último sorteio oficial.
- **⚙️ MLOps & Aprendizado Contínuo (`/mlops`):** Painel de avaliação automatizada da acurácia dos modelos e histórico de performance das gerações.
- **📊 Dashboard Interativo:** Estatísticas gerais de frequência de números, distribuição par/ímpar e análise por posição nas cartelas.
- **📈 Análise Histórica:** Consultas por período (semana, mês, ano) com métodos de cálculo bayesiano, frequência e padrões sequenciais.
- **🤖 Machine Learning:** Previsão de próximos sorteios utilizando modelo *RandomForest* treinado com pares sequenciais e distância de Hamming para diversidade de bilhetes.
- **🧭 Assistente JEV:** Interpreta o objetivo escrito pelo usuário e sugere uma estratégia existente. A confiança do JEV mede a classificação do pedido, não a chance de ganhar.
- **🔄 Sincronização CAIXA (`/sync-caixa`):** Atualização automática incremental a partir do serviço da CAIXA Econômica Federal.

---

## 🛠️ Arquitetura & Tecnologias Utilizadas

- **Arquitetura Modular:** Aplicação Flask refatorada com Blueprints (`routes/main.py`, `routes/data.py`, `routes/analysis.py`, `routes/ml.py`).
- **Camada de Serviços:** Regras de negócio e estatística isoladas em `services/stats_service.py`.
- **Backend:** Python 3.8+, Flask, MySQL, Flasgger (Swagger UI).
- **Machine Learning & Dados:** Scikit-Learn, Pandas, NumPy, SciPy.
- **Frontend Moderno:** HTML5, CSS3 com variáveis customizadas, Glassmorphism, Font Awesome Pro/Free, Bootstrap 5.3.
- **Infraestrutura:** Docker e Docker Compose com `.env` para gestão de segredos.

---

## 🚀 Como Executar

### Pré-requisitos
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/Ronickbr/Lotofacil.git
   cd Lotofacil
   ```

2. **Configuração de Variáveis de Ambiente:**
   Copie o arquivo de exemplo e crie o seu `.env`:
   ```bash
   cp .env.example .env
   ```

3. **Inicie a aplicação com Docker:**
   ```bash
   docker-compose up --build -d
   ```

4. **Acesse no seu navegador:**
   - **App Principal:** [http://localhost:5000](http://localhost:5000)
   - **Fábrica de Jogos:** [http://localhost:5000/smart-generate](http://localhost:5000/smart-generate)
   - **Documentação Swagger:** [http://localhost:5000/apidocs/](http://localhost:5000/apidocs/)
   - **phpMyAdmin:** [http://localhost:8081](http://localhost:8081)

---

## 🔒 Segurança

As credenciais do banco de dados e chave secreta do Flask utilizam o arquivo `.env` (ignorado pelo Git). Nunca envie o `.env` para o repositório público.

## JEV e avaliação do modelo

Para habilitar a sugestão de estratégia na Fábrica de Jogos, configure `TYPESAFE_API_KEY` no `.env` e reinicie a aplicação. A aplicação envia apenas o texto digitado pelo usuário para `POST https://api.typesafe.ai/v1/systemone`, com uma pergunta `choice` sobre as estratégias disponíveis. Sem chave ou em caso de indisponibilidade, o usuário continua a escolher a estratégia manualmente. Uma resposta com confiança menor que 0,6 exige escolha manual.

O JEV é usado para interpretar a intenção do usuário. Ele não prevê resultados de sorteios. O modelo local usa concursos anteriores para classificar dezenas, com treinamento, calibração e teste separados por concurso e em ordem cronológica. O teste é um bloco final reservado; não equivale a validação contínua de apostas futuras. As métricas incluem Brier score, acertos médios do cartão de 15 dezenas e referência aleatória exata: **9 acertos médios** e **Brier 0,24** para probabilidade uniforme de 0,6 por dezena. Um resultado superior nesse bloco isolado não prova vantagem futura. O novo treinamento substitui o modelo antigo, porque históricos de avaliação com janelas diferentes não permitem uma comparação direta para promoção.

Execute os testes sem acesso ao banco com `python -m unittest tests.test_analysis tests.test_jev_and_validation -v`. A suíte completa requer as dependências do `requirements.txt` e os serviços configurados para os testes de rotas. O painel de IA precisa de pelo menos 200 concursos válidos em ordem crescente de `concurso`.

---

## 🤝 Contribuindo

1. Faça um Fork do projeto
2. Crie sua Feature Branch (`git checkout -b feature/NovaFeature`)
3. Commit suas alterações (`git commit -m 'feat: Adiciona nova funcionalidade'`)
4. Push para a Branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request.

---
Desenvolvido com ☕ e focado em ciência de dados aplicada. Boa sorte nas apostas! 🍀
