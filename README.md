# 🍀 Sistema de Análise da Lotofácil

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)

Uma plataforma avançada para análise estatística, gestão de resultados e previsão inteligente de sorteios da Lotofácil utilizando Machine Learning.

---

## 📖 Sobre o Projeto

Este sistema foi desenvolvido para oferecer ferramentas completas de análise estatística dos resultados da Lotofácil. Ele combina análises probabilísticas tradicionais (como estatística bayesiana e detecção de padrões) com modelos de aprendizado de máquina para prever as melhores dezenas para os próximos sorteios.

## ✨ Funcionalidades Principais

- **📊 Dashboard Interativo:** Estatísticas gerais de frequência de números, distribuição de pares/ímpares e frequência por posição nas cartelas.
- **📈 Análise Histórica:** Consulte períodos específicos (semana, mês, ano) com diferentes métodos preditivos (frequência, bayesiano, padrões sequenciais e método combinado).
- **🤖 Machine Learning:** Previsão de próximos sorteios utilizando modelo RandomForest treinado com dados históricos em pares sequenciais.
- **💾 Gerenciamento de Dados:** Upload de novos resultados manuais ou importação em lote para manter a base de dados atualizada.
- **🎯 Geração de Jogos:** Sugestão inteligente de jogos (cartelas) com base nas probabilidades calculadas e no modelo de IA.

## 🛠️ Tecnologias Utilizadas

- **Backend:** Python, Flask (Arquitetura Modular em Blueprints)
- **Machine Learning & Dados:** Scikit-Learn, Pandas, NumPy
- **Documentação de API:** Flasgger (Swagger UI)
- **Banco de Dados:** MySQL
- **Infraestrutura:** Docker, Docker Compose
- **Frontend:** HTML5, CSS3, JavaScript (Templates Bootstrap)

## 🚀 Como Executar

### Pré-requisitos
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/lotofacil-analise.git
   cd lotofacil-analise
   ```

3. **Configuração de Variáveis de Ambiente:**
   Copie o arquivo de exemplo e crie o seu `.env`:
   ```bash
   cp .env.example .env
   ```
   *As credenciais padrão do banco já estão configuradas para rodar localmente.*

4. **Inicie os containers com Docker:**
   ```bash
   docker-compose up --build -d
   ```

5. **Acesse a aplicação:**
   - **App Principal:** [http://localhost:5000](http://localhost:5000)
   - **Documentação da API (Swagger):** [http://localhost:5000/apidocs/](http://localhost:5000/apidocs/)
   - **Banco de Dados (phpMyAdmin):** [http://localhost:8081](http://localhost:8081)

## 🧠 Como Funciona o Modelo Preditivo

O sistema utiliza um **RandomForestClassifier** para encontrar padrões sequenciais nos sorteios (ex: Sorteio N -> Sorteio N+1).
Antes de realizar previsões na tela de IA, é necessário treinar o modelo através da rota interna ou pelo painel do sistema, o que irá gerar o arquivo `lotofacil_model.pkl`.

> **Nota:** Para que a previsão e análise histórica funcionem corretamente e com maior precisão, é necessário popular o banco de dados com os resultados oficiais mais recentes da Caixa.

## 🔒 Segurança

**Aviso Importante para Produção:** As senhas e dados sensíveis não são mais armazenadas no repositório. O banco de dados agora utiliza o arquivo `.env` para carregar as credenciais:
- Sempre adicione seu arquivo `.env` ao `.gitignore`.
- Ocultar a interface do phpMyAdmin ou desabilitar o container para implantação em produção.

## 🤝 Contribuindo

Contribuições são muito bem-vindas! Se você tem alguma ideia para melhorar as análises, detectar novos padrões temporais ou otimizar o código:

1. Faça um Fork do projeto
2. Crie sua Feature Branch (`git checkout -b feature/NovaAnalise`)
3. Faça o Commit de suas mudanças (`git commit -m 'Add: Nova análise temporal'`)
4. Faça o Push para a Branch (`git push origin feature/NovaAnalise`)
5. Abra um Pull Request.

---
Desenvolvido com ☕ e focado em ciência de dados aplicada. Boa sorte nas apostas! 🍀