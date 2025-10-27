#!/usr/bin/env python3
"""
Simple FloorMind Integration
A lightweight integration that avoids PyTorch loading issues
"""

import os
import json
import subprocess
import sys
from PIL import Image
import base64
import io
from typing import Dict, Any, Optional
from datetime import datetime

class SimpleFloorMindGenerator:
    """
    Simple FloorMind generator that uses external scripts to avoid segfaults
    """
    
    def __init__(self, model_path: str = "google"):
        self.model_path = model_path
        self.config = self._load_config()
        self.is_available = self._check_availability()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        
        config_path = os.path.join(self.model_path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _check_availability(self) -> bool:
        """Check if model files are available"""
        
        required_files = [
            "model.safetensors",
            "tokenizer_config.json",
            "scheduler_config.json"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_path, file)):
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        return {
            "is_available": self.is_available,
            "model_path": self.model_path,
            "config": self.config,
            "resolution": self.config.get("resolution", 512),
            "model_name": self.config.get("model_name", "runwayml/stable-diffusion-v1-5")
        }
    
    def create_generation_script(self, description: str, output_path: str, **kwargs) -> str:
        """Create a Python script for generation"""
        
        script_content = f'''
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch
import sys
import os

try:
    # Load base model components
    base_model = "runwayml/stable-diffusion-v1-5"
    
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    
    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    
    print("Loading scheduler...")
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    
    # Load fine-tuned weights
    print("Loading fine-tuned weights...")
    state_dict = safetensors.torch.load_file("{self.model_path}/model.safetensors")
    unet.load_state_dict(state_dict)
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    # Move to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    print(f"Generating on {{device}}...")
    
    # Generate image
    generator = torch.Generator(device=device).manual_seed({kwargs.get('seed', 42)})
    
    with torch.no_grad():
        result = pipeline(
            prompt="{description}",
            width={kwargs.get('width', 512)},
            height={kwargs.get('height', 512)},
            num_inference_steps={kwargs.get('num_inference_steps', 20)},
            guidance_scale={kwargs.get('guidance_scale', 7.5)},
            generator=generator
        )
    
    # Save image
    result.images[0].save("{output_path}")
    print(f"Image saved to: {output_path}")
    
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        return script_content
    
    def generate_floor_plan(
        self,
        description: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a floor plan using external script
        
        Returns:
            Path to generated image
        """
        
        if not self.is_available:
            raise RuntimeError("Model files not available")
        
        # Create output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated_floor_plan_{timestamp}.png"
        
        # Create generation script
        script_content = self.create_generation_script(description, output_path, **kwargs)
        script_path = "temp_generation_script.py"
        
        try:
            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run script
            print(f"🎨 Generating: {description[:50]}...")
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Generation successful!")
                return output_path
            else:
                print(f"❌ Generation failed:")
                print(result.stderr)
                raise RuntimeError(f"Generation failed: {result.stderr}")
        
        finally:
            # Clean up script
            if os.path.exists(script_path):
                os.remove(script_path)
    
    def generate_with_metadata(
        self,
        description: str,
        output_dir: str = "generated_floor_plans",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate floor plan with metadata"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(output_dir, f"floor_plan_{timestamp}.png")
        
        # Generate image
        result_path = self.generate_floor_plan(description, image_path, **kwargs)
        
        # Create metadata
        metadata = {
            "description": description,
            "timestamp": timestamp,
            "image_path": result_path,
            "parameters": kwargs,
            "model_info": self.get_model_info()
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"floor_plan_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

def test_simple_integration():
    """Test the simple integration"""
    
    print("🧪 Testing Simple FloorMind Integration...")
    print("=" * 50)
    
    try:
        # Create generator
        generator = SimpleFloorMindGenerator()
        
        # Check availability
        if not generator.is_available:
            print("❌ Model files not available")
            return False
        
        print("✅ Model files available")
        
        # Show model info
        info = generator.get_model_info()
        print("\\n📊 Model Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test generation
        print("\\n🎨 Testing generation...")
        description = "Modern 2-bedroom apartment with open kitchen"
        
        metadata = generator.generate_with_metadata(
            description=description,
            seed=42,
            num_inference_steps=15,  # Faster for testing
            guidance_scale=7.5
        )
        
        print("✅ Generation completed!")
        print(f"📁 Image: {metadata['image_path']}")
        print(f"📄 Metadata: {metadata['image_path'].replace('.png', '_metadata.json')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_integration()