from flask import Flask
from config import Config
from extensions import mysql
from flasgger import Swagger
from extensions import mysql

def create_app():
    app = Flask(__name__)
    app.jinja_env.globals.update(zip=zip)
    app.config.from_object(Config)

    # Inicializar extensões
    mysql.init_app(app)
    
    app.config['SWAGGER'] = {
        'title': 'Lotofácil Analysis API',
        'uiversion': 3,
        'description': 'API para análise estatística e geração de jogos da Lotofácil'
    }
    Swagger(app)

    _db_initialized = False

    @app.before_request
    def setup_db():
        nonlocal _db_initialized
        if not _db_initialized:
            try:
                cur = mysql.connection.cursor()
                cur.execute("SELECT 1")
                mysql.connection.commit()
                cur.close()
                _db_initialized = True
            except Exception as e:
                print(f"Database setup failed: {e}")

    from datetime import datetime
    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}

    # Registrar Blueprints
    from routes.main import main_bp
    from routes.data import data_bp
    from routes.analysis import analysis_bp
    from routes.ml import ml_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(ml_bp)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
