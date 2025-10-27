#!/usr/bin/env python3
"""
FloorMind API Routes
Clean, organized API endpoints
"""

from flask import Blueprint, request, jsonify
import base64
import io
import os
from datetime import datetime
from PIL import Image
import logging
import traceback

# Import our model manager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ..core.model_manager import get_model_manager
except ImportError:
    # Fallback for when running as script
    from core.model_manager import get_model_manager

# Create blueprint
api_bp = Blueprint('api', __name__)

# Setup logging
logger = logging.getLogger(__name__)

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    
    model_manager = get_model_manager()
    
    return jsonify({
        "status": "healthy",
        "service": "FloorMind AI Backend",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.is_loaded,
        "version": "2.0.0"
    })

@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    
    model_manager = get_model_manager()
    
    return jsonify({
        "status": "success",
        "model_info": model_manager.get_model_info()
    })

@api_bp.route('/model/load', methods=['POST'])
def load_model():
    """Load the model on demand"""
    
    try:
        logger.info("🔄 Loading model via API request...")
        
        model_manager = get_model_manager()
        
        # Check if already loaded
        if model_manager.is_loaded:
            return jsonify({
                "status": "success",
                "message": "Model already loaded",
                "model_info": model_manager.get_model_info()
            })
        
        # Load the model
        success = model_manager.load_model()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model loaded successfully",
                "model_info": model_manager.get_model_info()
            })
        else:
            model_info = model_manager.get_model_info()
            return jsonify({
                "status": "error",
                "error": model_info.get("error", "Unknown error"),
                "model_info": model_info
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Model loading endpoint failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@api_bp.route('/model/unload', methods=['POST'])
def unload_model():
    """Unload the model to free memory"""
    
    try:
        model_manager = get_model_manager()
        model_manager.unload_model()
        
        return jsonify({
            "status": "success",
            "message": "Model unloaded successfully"
        })
        
    except Exception as e:
        logger.error(f"❌ Model unloading failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@api_bp.route('/generate', methods=['POST'])
def generate_floor_plan():
    """Generate a floor plan using the trained model"""
    
    model_manager = get_model_manager()
    
    if not model_manager.is_loaded:
        return jsonify({
            "error": "Model not loaded. Please load the model first using /model/load"
        }), 400
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "Description is required"}), 400
        
        description = data['description']
        
        # Generation parameters with validation
        params = {
            'width': min(max(data.get('width', 512), 256), 1024),
            'height': min(max(data.get('height', 512), 256), 1024),
            'num_inference_steps': min(max(data.get('steps', 20), 5), 50),
            'guidance_scale': min(max(data.get('guidance', 7.5), 1.0), 20.0),
            'seed': data.get('seed', None)
        }
        
        logger.info(f"🎨 Generating floor plan: {description[:50]}...")
        
        # Generate image
        generated_image = model_manager.generate_floor_plan(
            description=description,
            **params
        )
        
        # Convert to base64
        image_base64 = image_to_base64(generated_image)
        
        # Save if requested
        saved_path = None
        if data.get('save', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "outputs/generated"
            os.makedirs(save_dir, exist_ok=True)
            saved_path = os.path.join(save_dir, f"floor_plan_{timestamp}.png")
            generated_image.save(saved_path)
            logger.info(f"💾 Saved to: {saved_path}")
        
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
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/generate/variations', methods=['POST'])
def generate_variations():
    """Generate multiple variations of a floor plan"""
    
    model_manager = get_model_manager()
    
    if not model_manager.is_loaded:
        return jsonify({
            "error": "Model not loaded. Please load the model first using /model/load"
        }), 400
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "Description is required"}), 400
        
        description = data['description']
        num_variations = min(max(data.get('variations', 4), 1), 8)  # Limit variations
        
        # Generation parameters
        base_params = {
            'width': min(max(data.get('width', 512), 256), 1024),
            'height': min(max(data.get('height', 512), 256), 1024),
            'num_inference_steps': min(max(data.get('steps', 20), 5), 50),
            'guidance_scale': min(max(data.get('guidance', 7.5), 1.0), 20.0),
        }
        
        base_seed = data.get('seed', 42)
        
        logger.info(f"🎨 Generating {num_variations} variations: {description[:50]}...")
        
        # Generate variations
        variations = []
        
        for i in range(num_variations):
            # Use different seeds for variations
            seed = base_seed + i if base_seed else None
            
            generated_image = model_manager.generate_floor_plan(
                description=description,
                seed=seed,
                **base_params
            )
            
            image_base64 = image_to_base64(generated_image)
            variations.append({
                "variation": i + 1,
                "image": image_base64,
                "seed": seed
            })
        
        return jsonify({
            "status": "success",
            "description": description,
            "variations": variations,
            "parameters": {**base_params, "base_seed": base_seed},
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Variation generation failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/presets', methods=['GET'])
def get_presets():
    """Get predefined floor plan presets"""
    
    presets = {
        "residential": [
            "Modern 3-bedroom apartment with open kitchen and living room",
            "Cozy 2-bedroom house with separate dining area",
            "Spacious studio apartment with efficient layout",
            "Luxury 4-bedroom penthouse with master suite and balcony",
            "Traditional family home with garage and dining room",
            "Contemporary loft with industrial design elements",
            "Small 1-bedroom apartment with compact kitchen",
            "Large family house with 5 bedrooms and 3 bathrooms"
        ],
        "commercial": [
            "Small office space with reception area and meeting rooms",
            "Open-plan coworking space with flexible seating",
            "Retail store layout with customer area and storage",
            "Restaurant floor plan with dining area and kitchen",
            "Medical clinic with waiting room and examination rooms",
            "Gym layout with equipment area and changing rooms",
            "Coffee shop with seating area and service counter",
            "Hotel suite with bedroom, bathroom, and sitting area"
        ],
        "architectural_styles": [
            "Traditional colonial house floor plan",
            "Modern minimalist apartment layout", 
            "Contemporary open-concept design",
            "Classic Victorian house floor plan",
            "Scandinavian-style compact living",
            "Industrial loft conversion",
            "Mediterranean villa layout",
            "Japanese-inspired minimalist design"
        ]
    }
    
    return jsonify({
        "status": "success",
        "presets": presets
    })

@api_bp.route('/generate/batch', methods=['POST'])
def generate_batch():
    """Generate multiple floor plans from different descriptions"""
    
    model_manager = get_model_manager()
    
    if not model_manager.is_loaded:
        return jsonify({
            "error": "Model not loaded. Please load the model first using /model/load"
        }), 400
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'descriptions' not in data:
            return jsonify({"error": "Descriptions array is required"}), 400
        
        descriptions = data['descriptions']
        if not isinstance(descriptions, list) or len(descriptions) == 0:
            return jsonify({"error": "Descriptions must be a non-empty array"}), 400
        
        # Limit batch size
        descriptions = descriptions[:5]  # Max 5 at once
        
        # Generation parameters
        base_params = {
            'width': min(max(data.get('width', 512), 256), 1024),
            'height': min(max(data.get('height', 512), 256), 1024),
            'num_inference_steps': min(max(data.get('steps', 20), 5), 50),
            'guidance_scale': min(max(data.get('guidance', 7.5), 1.0), 20.0),
        }
        
        base_seed = data.get('seed', 42)
        
        logger.info(f"🎨 Generating batch of {len(descriptions)} floor plans...")
        
        # Generate batch
        results = []
        
        for i, description in enumerate(descriptions):
            try:
                seed = base_seed + i if base_seed else None
                
                generated_image = model_manager.generate_floor_plan(
                    description=description,
                    seed=seed,
                    **base_params
                )
                
                image_base64 = image_to_base64(generated_image)
                
                results.append({
                    "index": i,
                    "description": description,
                    "image": image_base64,
                    "seed": seed,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to generate for description {i}: {e}")
                results.append({
                    "index": i,
                    "description": description,
                    "error": str(e),
                    "status": "error"
                })
        
        return jsonify({
            "status": "success",
            "results": results,
            "parameters": base_params,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500