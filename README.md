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
   git clone https://github.com/seu-usuario/lotofacil.git
   cd lotofacil
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

---

## 🤝 Contribuindo

1. Faça um Fork do projeto
2. Crie sua Feature Branch (`git checkout -b feature/NovaFeature`)
3. Commit suas alterações (`git commit -m 'feat: Adiciona nova funcionalidade'`)
4. Push para a Branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request.

---
Desenvolvido com ☕ e focado em ciência de dados aplicada. Boa sorte nas apostas! 🍀
