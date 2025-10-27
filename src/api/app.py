#!/usr/bin/env python3
"""
FloorMind API Server
Restructured Flask application with proper organization
"""

from flask import Flask
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import routes
try:
    from .routes import api_bp
except ImportError:
    # Fallback for when running as script
    from routes import api_bp

def create_app():
    """Create and configure the Flask application"""
    
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(api_bp)  # Also register without prefix for backward compatibility
    
    # Root endpoint
    @app.route('/')
    def root():
        return {
            "service": "FloorMind AI Backend",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "health": "/health",
                "model_info": "/model/info",
                "model_load": "/model/load",
                "model_unload": "/model/unload",
                "generate": "/generate",
                "generate_variations": "/generate/variations",
                "generate_batch": "/generate/batch",
                "presets": "/presets"
            }
        }
    
    return app

def main():
    """Main application entry point"""
    
    print("🏠 FloorMind AI Backend Server v2.0")
    print("=" * 50)
    print("🔧 Restructured with proper organization")
    print("📦 Model loading on-demand for stability")
    print("🚀 Enhanced error handling and logging")
    
    print("\n📡 Available endpoints:")
    print("   GET  / - API information")
    print("   GET  /health - Health check")
    print("   GET  /model/info - Model information")
    print("   POST /model/load - Load model on demand")
    print("   POST /model/unload - Unload model")
    print("   POST /generate - Generate floor plan")
    print("   POST /generate/variations - Generate variations")
    print("   POST /generate/batch - Batch generation")
    print("   GET  /presets - Get predefined presets")
    
    print(f"\n🌐 Server starting on http://localhost:5001")
    print("💡 Frontend should connect to this URL")
    print("=" * 50)
    
    # Create and run app
    app = create_app()
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 FloorMind server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logging.error(f"Server error: {e}")

if __name__ == "__main__":
    main()