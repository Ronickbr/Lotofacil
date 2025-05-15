# Dockerfile
FROM python:3.10-slim

# Força o uso de IPv4
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# Instala dependências de sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libmariadb-dev \
    libmariadb-dev-compat \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte
COPY . .

# Comando para executar a aplicação
CMD ["flask", "run", "--host=0.0.0.0"]