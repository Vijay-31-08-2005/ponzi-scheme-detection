from flask import Flask, render_template
from app.routes import api_routes

def create_app():
    app = Flask(__name__)

    # Register API routes
    app.register_blueprint(api_routes, url_prefix="/api")

    # Homepage Route (renders index.html)
    @app.route("/")
    def home():
        return render_template("index.html")

    return app
