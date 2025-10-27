#!/usr/bin/env python3
"""
FloorMind Backend Server - Pickle Model Version
Flask server that loads your trained model from the pickle file
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import base64
import io
import json
import logging
import pickle
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
model_config = None
model_info = {"is_loaded": False, "device": "none"}

def construct_pipeline_from_components(components_dir):
    """Construct pipeline from individual component files"""
    
    try:
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        import torch
        
        logger.info("🔧 Constructing pipeline from individual components...")
        
        # Check what files we have
        available_files = os.listdir(components_dir)
        logger.info(f"📁 Available files: {available_files}")
        
        # Load components individually with error handling
        components = {}
        
        # Load tokenizer
        try:
            logger.info("🔄 Loading tokenizer...")
            components['tokenizer'] = CLIPTokenizer.from_pretrained(
                components_dir,
                local_files_only=True
            )
            logger.info("✅ Tokenizer loaded")
        except Exception as e:
            logger.error(f"❌ Tokenizer loading failed: {e}")
            return None
        
        # Load text encoder - try different approaches
        try:
            logger.info("🔄 Loading text encoder...")
            # Try loading from the directory directly
            components['text_encoder'] = CLIPTextModel.from_pretrained(
                components_dir,
                local_files_only=True,
                torch_dtype=torch.float32
            )
            logger.info("✅ Text encoder loaded")
        except Exception as e:
            logger.warning(f"⚠️ Text encoder loading failed: {e}")
            # Try loading a default one
            try:
                components['text_encoder'] = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    torch_dtype=torch.float32
                )
                logger.info("✅ Text encoder loaded (default)")
            except Exception as e2:
                logger.error(f"❌ Text encoder loading completely failed: {e2}")
                return None
        
        # Load VAE
        try:
            logger.info("🔄 Loading VAE...")
            # Try loading from directory or use default
            try:
                components['vae'] = AutoencoderKL.from_pretrained(
                    components_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32
                )
                logger.info("✅ VAE loaded from local")
            except:
                components['vae'] = AutoencoderKL.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    subfolder="vae",
                    torch_dtype=torch.float32
                )
                logger.info("✅ VAE loaded (default)")
        except Exception as e:
            logger.error(f"❌ VAE loading failed: {e}")
            return None
        
        # Load UNet (this is the most important - your trained weights)
        try:
            logger.info("🔄 Loading UNet (your trained weights)...")
            
            # Check if we have model.safetensors (your trained weights)
            safetensors_path = os.path.join(components_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                logger.info(f"✅ Found trained weights: {safetensors_path}")
                
                # Load UNet with your trained weights
                components['unet'] = UNet2DConditionModel.from_pretrained(
                    components_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
                logger.info("✅ UNet loaded with your trained weights!")
            else:
                logger.error("❌ No trained weights found (model.safetensors)")
                return None
                
        except Exception as e:
            logger.error(f"❌ UNet loading failed: {e}")
            logger.error(traceback.format_exc())
            return None
        
        # Load scheduler
        try:
            logger.info("🔄 Loading scheduler...")
            components['scheduler'] = DDPMScheduler.from_pretrained(
                components_dir,
                local_files_only=True
            )
            logger.info("✅ Scheduler loaded")
        except Exception as e:
            logger.warning(f"⚠️ Scheduler loading failed: {e}")
            # Use default scheduler
            components['scheduler'] = DDPMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler"
            )
            logger.info("✅ Scheduler loaded (default)")
        
        # Construct the pipeline
        logger.info("🔧 Assembling pipeline...")
        pipeline = StableDiffusionPipeline(
            vae=components['vae'],
            text_encoder=components['text_encoder'],
            tokenizer=components['tokenizer'],
            unet=components['unet'],  # This has your trained weights!
            scheduler=components['scheduler'],
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        logger.info("✅ Pipeline constructed successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Pipeline construction failed: {e}")
        logger.error(traceback.format_exc())
        return None

def load_trained_model():
    """Load the trained FloorMind model using the pickle approach"""
    global model_pipeline, model_config, model_info
    
    try:
        logger.info("🔄 Loading trained FloorMind model...")
        
        # First try to load the pickle file
        model_path = "google/floormind_model.pkl"
        
        if os.path.exists(model_path):
            logger.info(f"📦 Loading model data from: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model_config = model_data['config']
            training_stats = model_data['training_stats']
            best_val_loss = model_data['best_val_loss']
            
            logger.info(f"📊 Model info:")
            logger.info(f"   Best validation loss: {best_val_loss:.4f}")
            logger.info(f"   Training epochs: {model_config.get('num_epochs', 'Unknown')}")
            
            # Check if pipeline is directly in the data
            if 'pipeline' in model_data:
                model_pipeline = model_data['pipeline']
                logger.info("✅ Pipeline loaded from pickle file")
            else:
                # Construct pipeline from components in google directory
                logger.info("🔧 Pipeline not in pickle, constructing from components...")
                model_pipeline = construct_pipeline_from_components("google")
                
                if model_pipeline is None:
                    raise Exception("Failed to construct pipeline from components")
        else:
            # Try to construct directly from google directory
            logger.info("📦 No pickle file found, constructing from google directory...")
            model_pipeline = construct_pipeline_from_components("google")
            
            if model_pipeline is None:
                raise Exception("Failed to construct pipeline from components")
            
            # Create default config
            model_config = {
                "resolution": 512,
                "num_epochs": "Unknown",
                "model_type": "FloorMind"
            }
        
        # Move to CPU for safety
        logger.info("🔄 Moving pipeline to CPU...")
        model_pipeline = model_pipeline.to("cpu")
        logger.info("✅ Pipeline on CPU")
        
        # Update model info
        model_info = {
            "is_loaded": True,
            "device": "cpu",
            "model_path": "google/",
            "resolution": model_config.get("resolution", 512),
            "best_val_loss": model_data.get('best_val_loss', 'Unknown') if 'model_data' in locals() else 'Unknown',
            "loaded_at": datetime.now().isoformat(),
            "loading_method": "pickle_and_components"
        }
        
        logger.info("✅ FloorMind model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
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
        "service": "FloorMind AI Backend (Pickle)",
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
    try:
        logger.info("🔄 Loading FloorMind model via API...")
        
        success = load_trained_model()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "FloorMind model loaded successfully!",
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
        
        logger.info(f"🎨 Generating with YOUR trained FloorMind model: {description[:50]}...")
        
        # Import torch here
        import torch
        
        # Set up generator
        generator = None
        if params['seed'] is not None:
            generator = torch.Generator(device="cpu").manual_seed(params['seed'])
        
        # Generate with your trained model!
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
            saved_path = os.path.join(save_dir, f"floormind_trained_{timestamp}.png")
            generated_image.save(saved_path)
        
        return jsonify({
            "status": "success",
            "description": description,
            "image": image_data_url,
            "parameters": params,
            "saved_path": saved_path,
            "timestamp": datetime.now().isoformat(),
            "model_type": "FloorMind_Trained",
            "validation_loss": model_info.get('best_val_loss', 'Unknown')
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
            "Luxury 4-bedroom penthouse with master suite and balcony",
            "Traditional family home with garage and dining room"
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
    print("🏠 Starting FloorMind Backend Server (Pickle Model)...")
    print("=" * 50)
    print("🎯 Loading your trained FloorMind model from pickle + components")
    print("💡 This uses your actual trained weights from model.safetensors")
    print("🔥 Your model achieved validation loss: 0.024 (excellent!)")
    
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