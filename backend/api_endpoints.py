#!/usr/bin/env python3
"""
FloorMind API Endpoints
Flask API endpoints for floor plan generation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import base64
import io
from PIL import Image
import logging
from datetime import datetime
from typing import Dict, Any, List

# Import the FloorMind generator
from floormind_generator import FloorMindGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global generator instance
generator = None

def initialize_generator():
    """Initialize the FloorMind generator"""
    global generator
    
    try:
        logger.info("🔄 Initializing FloorMind generator...")
        generator = FloorMindGenerator(model_path="../google")
        logger.info("✅ FloorMind generator initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize generator: {e}")
        return False

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    
    return jsonify({
        "status": "healthy",
        "service": "FloorMind AI Generator",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": generator is not None and generator.is_loaded
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    
    if not generator:
        return jsonify({"error": "Generator not initialized"}), 500
    
    info = generator.get_model_info()
    return jsonify({
        "status": "success",
        "model_info": info
    })

@app.route('/generate', methods=['POST'])
def generate_floor_plan():
    """Generate a single floor plan"""
    
    if not generator or not generator.is_loaded:
        return jsonify({"error": "Generator not available"}), 500
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "Description is required"}), 400
        
        description = data['description']
        
        # Generation parameters
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 20),
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', None)
        }
        
        logger.info(f"🎨 Generating floor plan: {description[:50]}...")
        
        # Generate image
        image = generator.generate_floor_plan(description, **params)
        
        # Convert to base64 for JSON response
        image_base64 = image_to_base64(image)
        
        # Save if requested
        saved_path = None
        if data.get('save', False):
            saved_path = generator.save_generation(
                image, 
                description,
                metadata={
                    "api_request": True,
                    "parameters": params,
                    "client_ip": request.remote_addr
                }
            )
        
        return jsonify({
            "status": "success",
            "description": description,
            "image": image_base64,
            "parameters": params,
            "saved_path": saved_path,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate/variations', methods=['POST'])
def generate_variations():
    """Generate multiple variations of a floor plan"""
    
    if not generator or not generator.is_loaded:
        return jsonify({"error": "Generator not available"}), 500
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "Description is required"}), 400
        
        description = data['description']
        num_variations = data.get('variations', 4)
        
        # Generation parameters
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 20),
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', 42)
        }
        
        logger.info(f"🎨 Generating {num_variations} variations: {description[:50]}...")
        
        # Generate variations
        images = generator.generate_multiple_variations(
            description, 
            num_variations=num_variations,
            **params
        )
        
        # Convert all images to base64
        variations = []
        for i, image in enumerate(images):
            image_base64 = image_to_base64(image)
            variations.append({
                "variation": i + 1,
                "image": image_base64,
                "seed": params['seed'] + i
            })
        
        return jsonify({
            "status": "success",
            "description": description,
            "variations": variations,
            "parameters": params,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Variation generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate/batch', methods=['POST'])
def generate_batch():
    """Generate floor plans for multiple descriptions"""
    
    if not generator or not generator.is_loaded:
        return jsonify({"error": "Generator not available"}), 500
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'descriptions' not in data:
            return jsonify({"error": "Descriptions list is required"}), 400
        
        descriptions = data['descriptions']
        
        if not isinstance(descriptions, list) or len(descriptions) == 0:
            return jsonify({"error": "Descriptions must be a non-empty list"}), 400
        
        # Generation parameters
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 20),
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', 42)
        }
        
        logger.info(f"🎨 Batch generating {len(descriptions)} floor plans...")
        
        # Generate all images
        images = generator.batch_generate(descriptions, **params)
        
        # Convert all images to base64
        results = []
        for i, (description, image) in enumerate(zip(descriptions, images)):
            image_base64 = image_to_base64(image)
            results.append({
                "index": i,
                "description": description,
                "image": image_base64,
                "seed": params['seed'] + i
            })
        
        return jsonify({
            "status": "success",
            "results": results,
            "parameters": params,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/presets', methods=['GET'])
def get_presets():
    """Get predefined floor plan presets"""
    
    presets = {
        "residential": [
            "Modern 1-bedroom studio apartment with open layout",
            "Cozy 2-bedroom apartment with separate kitchen and living room",
            "Spacious 3-bedroom family apartment with master suite",
            "Luxury 4-bedroom penthouse with balcony and walk-in closets"
        ],
        "commercial": [
            "Small office space with reception area and meeting room",
            "Open-plan coworking space with flexible seating",
            "Retail store layout with customer area and storage",
            "Restaurant floor plan with dining area and kitchen"
        ],
        "architectural_styles": [
            "Traditional colonial house floor plan",
            "Modern minimalist apartment layout",
            "Contemporary open-concept design",
            "Classic Victorian house floor plan"
        ]
    }
    
    return jsonify({
        "status": "success",
        "presets": presets
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Initialize the generator when the module is imported
if __name__ == "__main__":
    print("🏠 Starting FloorMind API Server...")
    
    # Initialize generator
    if not initialize_generator():
        print("❌ Failed to initialize generator. Exiting.")
        sys.exit(1)
    
    # Start the server
    print("✅ FloorMind API Server ready!")
    print("📡 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /model/info - Model information")
    print("   POST /generate - Generate single floor plan")
    print("   POST /generate/variations - Generate variations")
    print("   POST /generate/batch - Batch generation")
    print("   GET  /presets - Get predefined presets")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )