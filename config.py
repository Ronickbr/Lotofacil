import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'sua_chave_secreta')
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'db')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'secret')
    MYSQL_DB = os.getenv('MYSQL_DB', 'lotofacil')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))