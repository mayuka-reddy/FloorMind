#!/usr/bin/env python3
"""
FloorMind Model Manager
Centralized model loading and management
"""

import os
import sys
import torch
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from PIL import Image
import traceback

class FloorMindModelManager:
    """
    Centralized model manager for FloorMind
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the model manager
        
        Args:
            model_path: Path to the trained model directory
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path or self._find_model_path()
        self.pipeline = None
        self.model_info = {}
        self.is_loaded = False
    
    def _find_model_path(self) -> str:
        """Find the model path automatically"""
        
        # Common model paths to check
        possible_paths = [
            "models/trained",
            "google",
            "../google",
            "../../google",
            os.path.join(os.path.dirname(__file__), "../../models/trained"),
            os.path.join(os.path.dirname(__file__), "../../google")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if it contains model files
                if self._validate_model_path(path):
                    self.logger.info(f"Found model at: {path}")
                    return path
        
        # Default fallback
        return "models/trained"
    
    def _validate_model_path(self, path: str) -> bool:
        """Validate that the path contains required model files"""
        
        required_files = [
            "model.safetensors",
            "tokenizer_config.json", 
            "scheduler_config.json"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False
        
        return True
    
    def load_model(self) -> bool:
        """Load the FloorMind model with comprehensive error handling"""
        
        try:
            self.logger.info("🔄 Loading FloorMind model...")
            
            # Validate model path
            if not os.path.exists(self.model_path):
                error_msg = f"Model path not found: {self.model_path}"
                self.logger.error(f"❌ {error_msg}")
                self._set_error_info(error_msg)
                return False
            
            # Validate model files
            if not self._validate_model_path(self.model_path):
                error_msg = f"Invalid model files in: {self.model_path}"
                self.logger.error(f"❌ {error_msg}")
                self._set_error_info(error_msg)
                return False
            
            # Import required libraries
            try:
                from diffusers import StableDiffusionPipeline
                import torch
            except ImportError as e:
                error_msg = f"Missing required libraries: {e}"
                self.logger.error(f"❌ {error_msg}")
                self._set_error_info(error_msg)
                return False
            
            # Load pipeline with multiple fallback strategies
            self.pipeline = self._load_pipeline_with_fallbacks()
            
            if self.pipeline is None:
                error_msg = "Failed to load pipeline with any method"
                self.logger.error(f"❌ {error_msg}")
                self._set_error_info(error_msg)
                return False
            
            # Configure device and optimizations
            self._configure_pipeline()
            
            # Set success info
            self.model_info = {
                "is_loaded": True,
                "device": str(self.pipeline.device),
                "model_path": self.model_path,
                "resolution": 512,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "loaded_at": datetime.now().isoformat()
            }
            
            self.is_loaded = True
            self.logger.info(f"✅ FloorMind model loaded successfully on {self.pipeline.device}")
            return True
            
        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.logger.error(traceback.format_exc())
            self._set_error_info(error_msg)
            return False
    
    def _load_pipeline_with_fallbacks(self):
        """Load pipeline with multiple fallback strategies"""
        
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Strategy 1: Direct loading with safetensors
        try:
            self.logger.info("📦 Attempting direct pipeline loading...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.logger.info("✅ Direct loading successful")
            return pipeline
            
        except Exception as e:
            self.logger.warning(f"⚠️ Direct loading failed: {e}")
        
        # Strategy 2: Component-based loading
        try:
            self.logger.info("📦 Attempting component-based loading...")
            pipeline = self._load_from_components()
            if pipeline:
                self.logger.info("✅ Component loading successful")
                return pipeline
                
        except Exception as e:
            self.logger.warning(f"⚠️ Component loading failed: {e}")
        
        # Strategy 3: Base model with fine-tuned weights
        try:
            self.logger.info("📦 Attempting base model + weights loading...")
            pipeline = self._load_base_with_weights()
            if pipeline:
                self.logger.info("✅ Base + weights loading successful")
                return pipeline
                
        except Exception as e:
            self.logger.warning(f"⚠️ Base + weights loading failed: {e}")
        
        return None
    
    def _load_from_components(self):
        """Load pipeline from individual components"""
        
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        import safetensors.torch
        
        base_model = "runwayml/stable-diffusion-v1-5"
        
        # Load components
        tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        scheduler = DDPMScheduler.from_pretrained(self.model_path, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        
        # Load fine-tuned weights
        safetensors_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = safetensors.torch.load_file(safetensors_path)
            unet.load_state_dict(state_dict, strict=False)
            self.logger.info("✅ Loaded fine-tuned UNet weights")
        
        # Create pipeline
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        return pipeline
    
    def _load_base_with_weights(self):
        """Load base model and apply fine-tuned weights"""
        
        from diffusers import StableDiffusionPipeline
        import torch
        import safetensors.torch
        
        base_model = "runwayml/stable-diffusion-v1-5"
        
        # Load base pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Apply fine-tuned weights if available
        safetensors_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = safetensors.torch.load_file(safetensors_path)
            pipeline.unet.load_state_dict(state_dict, strict=False)
            self.logger.info("✅ Applied fine-tuned weights to base model")
        
        return pipeline
    
    def _configure_pipeline(self):
        """Configure pipeline for optimal performance"""
        
        import torch
        
        # Move to appropriate device
        device = "cpu"  # Start with CPU for stability
        self.pipeline = self.pipeline.to(device)
        
        # Enable memory optimizations
        try:
            self.pipeline.enable_attention_slicing()
            self.logger.info("✅ Enabled attention slicing")
        except:
            pass
        
        try:
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                self.logger.info("✅ Enabled CPU offload")
        except:
            pass
    
    def _set_error_info(self, error_msg: str):
        """Set error information"""
        
        self.model_info = {
            "is_loaded": False,
            "error": error_msg,
            "device": "none",
            "failed_at": datetime.now().isoformat()
        }
        self.is_loaded = False
    
    def generate_floor_plan(
        self,
        description: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate a floor plan from description"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        # Set up generator for reproducible results
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = self.pipeline(
                prompt=description,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        return result.images[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()
    
    def unload_model(self):
        """Unload the model to free memory"""
        
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        self.is_loaded = False
        self.model_info = {"is_loaded": False}
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("✅ Model unloaded successfully")

# Global model manager instance
_model_manager = None

def get_model_manager(model_path: str = None) -> FloorMindModelManager:
    """Get the global model manager instance"""
    
    global _model_manager
    
    if _model_manager is None:
        _model_manager = FloorMindModelManager(model_path)
    
    return _model_manager

def load_model(model_path: str = None) -> bool:
    """Load the FloorMind model"""
    
    manager = get_model_manager(model_path)
    return manager.load_model()

def generate_floor_plan(description: str, **kwargs) -> Image.Image:
    """Generate a floor plan using the loaded model"""
    
    manager = get_model_manager()
    return manager.generate_floor_plan(description, **kwargs)

def get_model_info() -> Dict[str, Any]:
    """Get model information"""
    
    manager = get_model_manager()
    return manager.get_model_info()

def is_model_loaded() -> bool:
    """Check if model is loaded"""
    
    manager = get_model_manager()
    return manager.is_loaded