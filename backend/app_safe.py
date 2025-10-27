#!/usr/bin/env python3
"""
FloorMind Backend Server - Safe Loading Version
Flask server with conservative model loading approach
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
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_pipeline = None
model_info = {"is_loaded": False, "device": "none"}

def safe_import_libraries():
    """Safely import required libraries"""
    try:
        logger.info("🔄 Importing PyTorch...")
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} imported")
        
        logger.info("🔄 Importing Transformers...")
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__} imported")
        
        logger.info("🔄 Importing Diffusers...")
        import diffusers
        logger.info(f"✅ Diffusers {diffusers.__version__} imported")
        
        return torch, transformers, diffusers
    except Exception as e:
        logger.error(f"❌ Library import failed: {e}")
        return None, None, None

def load_model_components_separately():
    """Load model components one by one for better error isolation"""
    global model_pipeline, model_info
    
    try:
        torch, transformers, diffusers = safe_import_libraries()
        if not all([torch, transformers, diffusers]):
            raise Exception("Failed to import required libraries")
        
        model_path = "./google"
        
        logger.info("🔄 Loading tokenizer...")
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        logger.info("✅ Tokenizer loaded")
        
        logger.info("🔄 Loading text encoder...")
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=torch.float32
        )
        logger.info("✅ Text encoder loaded")
        
        logger.info("🔄 Loading VAE...")
        vae = diffusers.AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae", 
            local_files_only=True,
            torch_dtype=torch.float32
        )
        logger.info("✅ VAE loaded")
        
        logger.info("🔄 Loading UNet...")
        unet = diffusers.UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            local_files_only=True,
            torch_dtype=torch.float32
        )
        logger.info("✅ UNet loaded")
        
        logger.info("🔄 Loading scheduler...")
        scheduler = diffusers.DDPMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=True
        )
        logger.info("✅ Scheduler loaded")
        
        logger.info("🔄 Assembling pipeline...")
        model_pipeline = diffusers.StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        logger.info("✅ Pipeline assembled")
        
        # Move to CPU for safety
        logger.info("🔄 Moving pipeline to CPU...")
        model_pipeline = model_pipeline.to("cpu")
        logger.info("✅ Pipeline on CPU")
        
        model_info = {
            "is_loaded": True,
            "device": "cpu",
            "model_path": model_path,
            "resolution": 512,
            "torch_version": torch.__version__,
            "loaded_at": datetime.now().isoformat(),
            "loading_method": "component_by_component"
        }
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component loading failed: {e}")
        logger.error(traceback.format_exc())
        model_info = {
            "is_loaded": False,
            "error": str(e),
            "device": "none"
        }
        return False

def load_model_direct():
    """Try direct pipeline loading with minimal settings"""
    global model_pipeline, model_info
    
    try:
        torch, transformers, diffusers = safe_import_libraries()
        if not all([torch, transformers, diffusers]):
            raise Exception("Failed to import required libraries")
        
        model_path = "./google"
        
        logger.info("🔄 Loading pipeline directly...")
        
        # Try with most conservative settings
        model_pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
            use_safetensors=True,
            variant=None
        )
        
        logger.info("✅ Pipeline loaded directly")
        
        # Move to CPU
        model_pipeline = model_pipeline.to("cpu")
        logger.info("✅ Pipeline on CPU")
        
        model_info = {
            "is_loaded": True,
            "device": "cpu", 
            "model_path": model_path,
            "resolution": 512,
            "torch_version": torch.__version__,
            "loaded_at": datetime.now().isoformat(),
            "loading_method": "direct"
        }
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct loading failed: {e}")
        logger.error(traceback.format_exc())
        model_info = {
            "is_loaded": False,
            "error": str(e),
            "device": "none"
        }
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "FloorMind AI Backend (Safe)",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_pipeline is not None
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    return jsonify({
        "status": "success",
        "model_info": model_info
    })

@app.route('/model/load', methods=['POST'])
def load_model_endpoint():
    global model_pipeline, model_info
    
    try:
        data = request.get_json() or {}
        method = data.get('method', 'direct')  # 'direct' or 'components'
        
        logger.info(f"🔄 Loading model using {method} method...")
        
        if method == 'components':
            success = load_model_components_separately()
        else:
            success = load_model_direct()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model loaded successfully",
                "model_info": model_info
            })
        else:
            return jsonify({
                "status": "failed",
                "error": model_info.get('error', 'Unknown error'),
                "model_info": model_info
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Model loading endpoint failed: {e}")
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate_floor_plan():
    if model_pipeline is None:
        return jsonify({
            "error": "Model not loaded. Please load the model first.",
            "status": "model_not_loaded"
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "Description is required"}), 400
        
        description = data['description']
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 20),
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', None)
        }
        
        logger.info(f"🎨 Generating with trained model: {description[:50]}...")
        
        # Import torch here to avoid issues during startup
        import torch
        
        # Set up generator
        generator = None
        if params['seed'] is not None:
            generator = torch.Generator(device="cpu").manual_seed(params['seed'])
        
        # Generate with your trained model
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
        
        # Save if requested
        saved_path = None
        if data.get('save', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "./generated_floor_plans"
            os.makedirs(save_dir, exist_ok=True)
            saved_path = os.path.join(save_dir, f"trained_floor_plan_{timestamp}.png")
            generated_image.save(saved_path)
        
        return jsonify({
            "status": "success",
            "description": description,
            "image": image_data_url,
            "parameters": params,
            "saved_path": saved_path,
            "timestamp": datetime.now().isoformat(),
            "model_type": "trained"
        })
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/presets', methods=['GET'])
def get_presets():
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
            "Retail store layout with customer area and storage"
        ]
    }
    
    return jsonify({
        "status": "success",
        "presets": presets
    })

if __name__ == "__main__":
    print("🏠 Starting FloorMind Backend Server (Safe Loading)...")
    print("=" * 50)
    print("🔒 Using conservative model loading approach")
    print("💡 Use POST /model/load to load your trained model")
    print("🎯 Supports both 'direct' and 'components' loading methods")
    
    print(f"\n🚀 Server starting on http://localhost:5001")
    print("=" * 50)
    
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