#!/usr/bin/env python3
"""
FloorMind Backend Server - Minimal Version
Flask server that starts without loading the model initially
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import base64
import io
import json
import logging
from datetime import datetime
from PIL import Image
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_pipeline = None
model_info = {"is_loaded": False, "device": "none"}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "FloorMind AI Backend",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_pipeline is not None
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify({
        "status": "success",
        "model_info": model_info
    })

@app.route('/model/load', methods=['POST'])
def load_model_endpoint():
    """Load the model on demand"""
    global model_pipeline, model_info
    
    try:
        logger.info("🔄 Loading FloorMind model on demand...")
        
        # Check if model path exists
        model_path = "./google"
        
        if not os.path.exists(model_path):
            return jsonify({
                "error": f"Model path not found: {model_path}",
                "status": "failed"
            }), 404
        
        # Try to import and load
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Load the pipeline
        model_pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        
        # Use CPU for now to avoid CUDA issues
        device = "cpu"
        model_pipeline = model_pipeline.to(device)
        
        model_info = {
            "is_loaded": True,
            "device": device,
            "model_path": model_path,
            "resolution": 512,
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ FloorMind model loaded successfully on {device}")
        
        return jsonify({
            "status": "success",
            "message": "Model loaded successfully",
            "model_info": model_info
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model_info = {
            "is_loaded": False,
            "error": str(e),
            "device": "none"
        }
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/generate', methods=['POST'])
def generate_floor_plan():
    """Generate a floor plan using the trained model"""
    
    if model_pipeline is None:
        return jsonify({
            "error": "Model not loaded. Please load the model first using /model/load endpoint.",
            "status": "model_not_loaded"
        }), 400
    
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
        
        # Set up generator for reproducible results
        import torch
        generator = None
        if params['seed'] is not None:
            device = model_pipeline.device
            generator = torch.Generator(device=device).manual_seed(params['seed'])
        
        # Generate image
        with torch.no_grad():
            result = model_pipeline(
                prompt=description,
                width=params['width'],
                height=params['height'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                generator=generator
            )
        
        # Convert to base64
        generated_image = result.images[0]
        buffer = io.BytesIO()
        generated_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        return jsonify({
            "status": "success",
            "description": description,
            "image": image_data_url,
            "parameters": params,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/presets', methods=['GET'])
def get_presets():
    """Get predefined floor plan presets"""
    
    presets = {
        "residential": [
            "Modern 3-bedroom apartment with open kitchen and living room",
            "Cozy 2-bedroom house with separate dining area",
            "Spacious studio apartment with efficient layout",
            "Luxury 4-bedroom penthouse with master suite and balcony"
        ],
        "commercial": [
            "Small office space with reception area and meeting rooms",
            "Open-plan coworking space with flexible seating",
            "Retail store layout with customer area and storage",
            "Restaurant floor plan with dining area and kitchen"
        ]
    }
    
    return jsonify({
        "status": "success",
        "presets": presets
    })

if __name__ == "__main__":
    print("🏠 Starting FloorMind Backend Server (Minimal Mode)...")
    print("=" * 50)
    print("⚠️  Model will be loaded on-demand to avoid startup issues")
    print("💡 Use POST /model/load to load the model when ready")
    
    print("\n📡 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /model/info - Model information")
    print("   POST /model/load - Load model on demand")
    print("   POST /generate - Generate floor plan")
    print("   GET  /presets - Get predefined presets")
    
    print(f"\n🚀 Server starting on http://localhost:5001")
    print("=" * 50)
    
    # Start the server
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