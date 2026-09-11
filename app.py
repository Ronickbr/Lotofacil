from flask import Flask
from config import Config
from extensions import mysql
from flasgger import Swagger


def create_app(config_override=None):
    app = Flask(__name__)
    app.jinja_env.globals.update(zip=zip)
    app.config.from_object(Config)
    app.config.setdefault('INIT_DB_ON_REQUEST', True)
    if config_override:
        app.config.update(config_override)

    # Inicializar extensões
    mysql.init_app(app)
    
    app.config['SWAGGER'] = {
        'title': 'Lotofácil Analysis API',
        'uiversion': 3,
        'description': 'API para análise estatística e geração de jogos da Lotofácil'
    }
    Swagger(app)

    db_state = {'initialized': False}

    @app.before_request
    def setup_db():
        if not app.config['INIT_DB_ON_REQUEST'] or db_state['initialized']:
            return
        if not db_state['initialized']:
            try:
                cur = mysql.connection.cursor()
                cur.execute("SELECT 1")
                mysql.connection.commit()
                cur.close()
                db_state['initialized'] = True
            except Exception as e:
                app.logger.warning("Database setup failed: %s", e)

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
