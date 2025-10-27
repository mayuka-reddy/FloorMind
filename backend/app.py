#!/usr/bin/env python3
"""
FloorMind Backend Server
Flask server to serve the trained FloorMind model
"""

from flask import Flask, request, jsonify, send_file
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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_pipeline = None
model_info = {}

def load_model():
    """Load the trained FloorMind model with safe error handling"""
    global model_pipeline, model_info
    
    try:
        logger.info("🔄 Loading FloorMind model...")
        
        # Check if model path exists
        model_path = "../models/trained_model"  # Updated path after reorganization
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model path not found: {model_path}")
            logger.info("💡 Looking for alternative model locations...")
            
            # Try alternative paths
            alternative_paths = [
                "../google",
                "./models/trained_model",
                "../outputs/models/floormind_baseline"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    logger.info(f"✅ Found model at: {model_path}")
                    break
            else:
                model_info = {
                    "is_loaded": False,
                    "error": f"Model not found in any expected location",
                    "searched_paths": [model_path] + alternative_paths,
                    "device": "none"
                }
                return False
        
        # Check for required model files
        required_files = ["model.safetensors", "tokenizer_config.json", "scheduler_config.json"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing model files: {missing_files}")
            model_info = {
                "is_loaded": False,
                "error": f"Missing model files: {missing_files}",
                "device": "none"
            }
            return False
        
        # Try to import required libraries
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except ImportError as e:
            logger.error(f"❌ Missing required libraries: {e}")
            model_info = {
                "is_loaded": False,
                "error": f"Missing required libraries: {e}",
                "device": "none"
            }
            return False
        
        # Load the pipeline with safe settings and error handling
        logger.info("📦 Loading Stable Diffusion pipeline...")
        
        # Use safer loading approach
        try:
            # First try with low memory usage
            model_pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use float32 for stability
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,  # Only use local files
                low_cpu_mem_usage=True,  # Reduce memory usage
                use_safetensors=True  # Use safetensors format
            )
        except Exception as load_error:
            logger.warning(f"⚠️ Standard loading failed: {load_error}")
            # Fallback to component loading
            try:
                from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
                from transformers import CLIPTextModel, CLIPTokenizer
                
                base_model = "runwayml/stable-diffusion-v1-5"
                
                # Load components separately
                tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
                text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
                vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
                scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
                unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
                
                # Load fine-tuned weights
                import safetensors.torch
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    state_dict = safetensors.torch.load_file(safetensors_path)
                    unet.load_state_dict(state_dict, strict=False)
                    logger.info("✅ Loaded fine-tuned weights")
                
                # Create pipeline
                model_pipeline = StableDiffusionPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                    safety_checker=None,
                    feature_extractor=None
                )
                
            except Exception as component_error:
                logger.error(f"❌ Component loading also failed: {component_error}")
                raise component_error
        
        # Move to CPU first for safety
        device = "cpu"  # Start with CPU to avoid issues
        model_pipeline = model_pipeline.to(device)
        
        # Enable memory efficient attention if available
        try:
            model_pipeline.enable_attention_slicing()
            logger.info("✅ Enabled attention slicing for memory efficiency")
        except:
            pass
        
        # Store model info
        model_info = {
            "is_loaded": True,
            "device": device,
            "model_path": model_path,
            "resolution": 512,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ FloorMind model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        
        # Set fallback info
        model_info = {
            "is_loaded": False,
            "error": str(e),
            "device": "none",
            "failed_at": datetime.now().isoformat()
        }
        return False

def image_to_base64(image):
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
        logger.info("🔄 Loading model via API request...")
        
        if model_pipeline is not None:
            return jsonify({
                "status": "success",
                "message": "Model already loaded",
                "model_info": model_info
            })
        
        success = load_model()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model loaded successfully",
                "model_info": model_info
            })
        else:
            return jsonify({
                "status": "error",
                "error": model_info.get("error", "Unknown error"),
                "model_info": model_info
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Model loading endpoint failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate_floor_plan():
    """Generate a floor plan using the trained model"""
    
    if model_pipeline is None:
        return jsonify({
            "error": "Model not loaded. Please check server logs."
        }), 500
    
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
        image_base64 = image_to_base64(generated_image)
        
        # Save if requested
        saved_path = None
        if data.get('save', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "../generated_floor_plans"
            os.makedirs(save_dir, exist_ok=True)
            saved_path = os.path.join(save_dir, f"floor_plan_{timestamp}.png")
            generated_image.save(saved_path)
        
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

@app.route('/generate/variations', methods=['POST'])
def generate_variations():
    """Generate multiple variations of a floor plan"""
    
    if model_pipeline is None:
        return jsonify({
            "error": "Model not loaded. Please check server logs."
        }), 500
    
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
        variations = []
        import torch
        
        for i in range(num_variations):
            # Use different seeds for variations
            seed = params['seed'] + i
            generator = torch.Generator(device=model_pipeline.device).manual_seed(seed)
            
            with torch.no_grad():
                result = model_pipeline(
                    prompt=description,
                    width=params['width'],
                    height=params['height'],
                    num_inference_steps=params['num_inference_steps'],
                    guidance_scale=params['guidance_scale'],
                    generator=generator
                )
            
            image_base64 = image_to_base64(result.images[0])
            variations.append({
                "variation": i + 1,
                "image": image_base64,
                "seed": seed
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

@app.route('/presets', methods=['GET'])
def get_presets():
    """Get predefined floor plan presets"""
    
    presets = {
        "residential": [
            "Modern 3-bedroom apartment with open kitchen and living room",
            "Cozy 2-bedroom house with separate dining area",
            "Spacious studio apartment with efficient layout",
            "Luxury 4-bedroom penthouse with master suite and balcony",
            "Traditional family home with garage and dining room",
            "Contemporary loft with industrial design elements"
        ],
        "commercial": [
            "Small office space with reception area and meeting rooms",
            "Open-plan coworking space with flexible seating",
            "Retail store layout with customer area and storage",
            "Restaurant floor plan with dining area and kitchen",
            "Medical clinic with waiting room and examination rooms",
            "Gym layout with equipment area and changing rooms"
        ],
        "architectural_styles": [
            "Traditional colonial house floor plan",
            "Modern minimalist apartment layout", 
            "Contemporary open-concept design",
            "Classic Victorian house floor plan",
            "Scandinavian-style compact living",
            "Industrial loft conversion"
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

if __name__ == "__main__":
    print("🏠 Starting FloorMind Backend Server...")
    print("=" * 50)
    
    # Don't auto-load model to avoid startup crashes
    print("⚠️  Model will be loaded on demand via /model/load endpoint")
    print("💡 This prevents startup crashes and allows safer model loading")
    
    print("\n📡 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /model/info - Model information")
    print("   POST /model/load - Load model on demand")
    print("   POST /generate - Generate floor plan")
    print("   POST /generate/variations - Generate variations")
    print("   GET  /presets - Get predefined presets")
    
    print(f"\n🚀 Server starting on http://localhost:5001")
    print("=" * 50)
    
    # Start the server
    try:
        app.run(
            host='0.0.0.0',
            port=5001,  # Use port 5001 to avoid macOS AirPlay conflict
            debug=False,  # Set to False for production
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 FloorMind server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logger.error(traceback.format_exc())