from flask_mysqldb import MySQL

def init_db(app):
    # Configurações do MySQL
    app.config['MYSQL_HOST'] = 'db'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'secret'
    app.config['MYSQL_DB'] = 'lotofacil'
    app.config['MYSQL_PORT'] = 3306

    # Inicializa o objeto MySQL
    mysql = MySQL(app)
    return mysql