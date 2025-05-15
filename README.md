# Sistema de Análise da Lotofácil

## Sobre o Projeto
Este sistema oferece ferramentas avançadas para análise estatística dos resultados da Lotofácil, incluindo:
- Dashboard com estatísticas gerais
- Upload e gerenciamento de resultados de concursos
- Análise histórica de números sorteados
- Previsão de números com aprendizado de máquina
- Geração de estatísticas personalizadas

## Requisitos
- Docker
- Docker Compose

## Como executar
1. Clone este repositório
2. Execute `docker-compose up --build`
3. Acesse http://localhost:5000 para o aplicativo principal
4. Acesse http://localhost:8081 para o PHPMyAdmin (usuário: root, senha: secret)

## Funcionalidades

### Upload de Resultados
Adicione novos resultados de concursos manualmente ou faça upload de arquivos.

### Dashboard
Visualize estatísticas gerais, frequência de números e padrões de sorteio.

### Análise Histórica
Consulte estatísticas detalhadas de concursos anteriores e identifique tendências.

### Previsão de Números
Utilize o modelo de machine learning para gerar sugestões de apostas baseadas em padrões históricos.

## Segurança
Para ambientes de produção, recomenda-se alterar as senhas padrão no arquivo docker-compose.yml e configurar variáveis de ambiente adequadas.