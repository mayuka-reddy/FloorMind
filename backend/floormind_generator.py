#!/usr/bin/env python3
"""
FloorMind AI Generator
Integration module for the trained FloorMind model
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FloorMindGenerator:
    """
    FloorMind AI Generator for floor plan generation
    """
    
    def __init__(self, model_path: str = "google", device: Optional[str] = None):
        """
        Initialize the FloorMind generator
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.config = None
        self.is_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model pipeline"""
        
        try:
            self.logger.info(f"Loading FloorMind model from: {self.model_path}")
            
            # Try different loading methods
            if self._load_from_diffusers():
                self.logger.info("✅ Model loaded using diffusers format")
            elif self._load_from_components():
                self.logger.info("✅ Model loaded from individual components")
            else:
                raise ValueError("Could not load model using any method")
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            self.is_loaded = True
            
            self.logger.info(f"✅ FloorMind generator ready on {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
    
    def _load_from_diffusers(self) -> bool:
        """Try to load using standard diffusers format"""
        
        try:
            from diffusers import StableDiffusionPipeline
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not load with diffusers format: {e}")
            return False
    
    def _load_from_components(self) -> bool:
        """Load from individual component files"""
        
        try:
            from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            import safetensors.torch
            
            base_model = "runwayml/stable-diffusion-v1-5"
            
            # Load base components
            tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
            scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
            
            # Load UNet and fine-tuned weights
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
            
            # Load fine-tuned weights if available
            safetensors_path = os.path.join(self.model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = safetensors.torch.load_file(safetensors_path)
                unet.load_state_dict(state_dict)
                self.logger.info("✅ Loaded fine-tuned UNet weights")
            
            # Create pipeline
            self.pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Could not load from components: {e}")
            return False
    
    def generate_floor_plan(
        self,
        description: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate a floor plan from text description
        
        Args:
            description: Text description of the desired floor plan
            width: Output image width
            height: Output image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducible results
            **kwargs: Additional generation parameters
        
        Returns:
            PIL Image of the generated floor plan
        """
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please check initialization.")
        
        try:
            # Set up generator for reproducible results
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=description,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs
                )
            
            return result.images[0]
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def generate_multiple_variations(
        self,
        description: str,
        num_variations: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple variations of a floor plan
        
        Args:
            description: Text description of the desired floor plan
            num_variations: Number of variations to generate
            **kwargs: Additional generation parameters
        
        Returns:
            List of PIL Images
        """
        
        variations = []
        for i in range(num_variations):
            # Use different seeds for variations
            seed = kwargs.get('seed', 42) + i if 'seed' in kwargs else None
            kwargs_copy = kwargs.copy()
            kwargs_copy['seed'] = seed
            
            image = self.generate_floor_plan(description, **kwargs_copy)
            variations.append(image)
        
        return variations
    
    def batch_generate(
        self,
        descriptions: List[str],
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate floor plans for multiple descriptions
        
        Args:
            descriptions: List of text descriptions
            **kwargs: Generation parameters
        
        Returns:
            List of generated images
        """
        
        results = []
        for i, description in enumerate(descriptions):
            self.logger.info(f"Generating {i+1}/{len(descriptions)}: {description[:50]}...")
            
            # Use different seeds for each description
            seed = kwargs.get('seed', 42) + i if 'seed' in kwargs else None
            kwargs_copy = kwargs.copy()
            kwargs_copy['seed'] = seed
            
            image = self.generate_floor_plan(description, **kwargs_copy)
            results.append(image)
        
        return results
    
    def save_generation(
        self,
        image: Image.Image,
        description: str,
        output_dir: str = "generated_floor_plans",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save generated floor plan with metadata
        
        Args:
            image: Generated image
            description: Original description
            output_dir: Output directory
            metadata: Additional metadata to save
        
        Returns:
            Path to saved image
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"floor_plan_{timestamp}.png"
        image_path = os.path.join(output_dir, filename)
        
        # Save image
        image.save(image_path)
        
        # Save metadata
        metadata_dict = {
            "description": description,
            "timestamp": timestamp,
            "image_path": image_path,
            "model_device": self.device,
            **(metadata or {})
        }
        
        metadata_path = os.path.join(output_dir, f"floor_plan_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        self.logger.info(f"✅ Saved: {image_path}")
        return image_path
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        return {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "model_path": self.model_path,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "pipeline_type": type(self.pipeline).__name__ if self.pipeline else None
        }

# Convenience functions for easy integration
def create_generator(model_path: str = "google") -> FloorMindGenerator:
    """Create a FloorMind generator instance"""
    return FloorMindGenerator(model_path)

def generate_floor_plan(description: str, model_path: str = "google", **kwargs) -> Image.Image:
    """Quick function to generate a single floor plan"""
    generator = FloorMindGenerator(model_path)
    return generator.generate_floor_plan(description, **kwargs)

# Example usage
if __name__ == "__main__":
    # Test the generator
    print("🏠 Testing FloorMind Generator...")
    
    try:
        # Create generator
        generator = FloorMindGenerator()
        
        # Test generation
        test_descriptions = [
            "Modern 3-bedroom apartment with open kitchen and living room",
            "Traditional 2-bedroom house with separate dining room",
            "Studio apartment with efficient space utilization"
        ]
        
        for i, description in enumerate(test_descriptions):
            print(f"\\nGenerating: {description}")
            image = generator.generate_floor_plan(description, seed=42+i)
            
            # Save the result
            output_path = generator.save_generation(
                image, 
                description,
                metadata={"test_run": True, "variation": i}
            )
            print(f"Saved to: {output_path}")
        
        print("\\n✅ FloorMind Generator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()