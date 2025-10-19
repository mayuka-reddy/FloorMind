"""
FloorMind Backend API
Main Flask application for serving floor plan generation requests
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routes.generate import generate_bp

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = '../outputs/sample_generations'
    
    # Ensure output directories exist
    os.makedirs('../outputs/sample_generations', exist_ok=True)
    os.makedirs('../outputs/metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(generate_bp, url_prefix='/api')
    
    # Import and register evaluate blueprint
    try:
        from routes.evaluate import evaluate_bp
        app.register_blueprint(evaluate_bp, url_prefix='/api')
    except ImportError:
        print("Warning: Could not import evaluate blueprint")
    
    @app.route('/')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'FloorMind API',
            'version': '1.0.0'
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("Starting FloorMind API server...")
    print("Available endpoints:")
    print("  GET  /           - Health check")
    print("  POST /api/generate - Generate floor plan")
    print("  GET  /api/evaluate - Get model metrics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)