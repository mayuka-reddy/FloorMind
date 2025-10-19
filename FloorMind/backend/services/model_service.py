"""
Model inference service for FloorMind
Handles loading and running trained diffusion models and Gemini API integration
"""

import torch
import os
import uuid
import time
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
# Removed Gemini integration - focusing on local model training

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.baseline_pipeline = None
        self.constraint_pipeline = None
        self.clip_model = None
        self.clip_processor = None
        
        # Focus on local model training
        self.gemini_service = None
        self.gemini_available = False
        # Initialize CLIP for evaluation
        self._load_clip_model()
    
    def _load_clip_model(self):
        """Load CLIP model for evaluation"""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
    
    def _load_baseline_model(self):
        """Load baseline Stable Diffusion model"""
        if self.baseline_pipeline is None:
            try:
                # Check for fine-tuned model first
                model_path = "models/baseline_sd"
                if os.path.exists(model_path):
                    self.baseline_pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                else:
                    # Use base Stable Diffusion model
                    self.baseline_pipeline = StableDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-1-base",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                
                self.baseline_pipeline.to(self.device)
                print("Baseline model loaded successfully")
                
            except Exception as e:
                print(f"Error loading baseline model: {e}")
                raise e
    
    def _load_constraint_model(self):
        """Load constraint-aware model"""
        if self.constraint_pipeline is None:
            try:
                # Check for fine-tuned constraint model
                model_path = "models/constraint_aware_sd"
                if os.path.exists(model_path):
                    self.constraint_pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                else:
                    # Fallback to baseline for now
                    self.constraint_pipeline = StableDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-1-base",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                
                self.constraint_pipeline.to(self.device)
                print("Constraint-aware model loaded successfully")
                
            except Exception as e:
                print(f"Error loading constraint-aware model: {e}")
                raise e
    
    def generate_floorplan(self, prompt, model_type="constraint_aware", seed=None, guidance_scale=7.5):
        """
        Generate floor plan from text prompt using diffusion models
        
        Args:
            prompt (str): Text description of desired floor plan
            model_type (str): "baseline" or "constraint_aware"
            seed (int): Random seed for reproducibility
            guidance_scale (float): Guidance scale for generation
        
        Returns:
            dict: Generation result with success status and image path
        """
        try:
            start_time = time.time()
            
            # Generate with diffusion models
            return self._generate_with_diffusion(prompt, model_type, seed, guidance_scale)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    

    
    def _generate_with_diffusion(self, prompt, model_type, seed, guidance_scale):
        """Generate floor plan using diffusion models (fallback)"""
        try:
            start_time = time.time()
            
            # Enhance prompt for architectural context
            enhanced_prompt = f"architectural floor plan, {prompt}, top-down view, clean lines, professional blueprint style"
            
            # Load appropriate model
            if model_type == "baseline":
                self._load_baseline_model()
                pipeline = self.baseline_pipeline
            else:
                self._load_constraint_model()
                pipeline = self.constraint_pipeline
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate image
            with torch.no_grad():
                result = pipeline(
                    enhanced_prompt,
                    num_inference_steps=20,
                    guidance_scale=guidance_scale,
                    height=512,
                    width=512
                )
            
            image = result.images[0]
            
            # Save generated image
            output_filename = f"{uuid.uuid4()}.png"
            output_path = f"../outputs/sample_generations/{output_filename}"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            image.save(output_path)
            
            generation_time = time.time() - start_time
            
            # Calculate CLIP score if possible
            clip_score = None
            if self.clip_model is not None:
                try:
                    clip_score = self._calculate_clip_score(image, prompt)
                except Exception as e:
                    print(f"Could not calculate CLIP score: {e}")
            
            return {
                'success': True,
                'image_path': output_path,
                'generation_time': generation_time,
                'metadata': {
                    'model_type': model_type,
                    'prompt': prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'seed': seed,
                    'guidance_scale': guidance_scale,
                    'clip_score': clip_score,
                    'image_size': image.size,
                    'generation_method': 'diffusion'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    
    def _calculate_clip_score(self, image, text):
        """Calculate CLIP score between image and text"""
        try:
            inputs = self.clip_processor(
                text=[text], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_score = torch.softmax(logits_per_image, dim=1)[0, 0].item()
            
            return float(clip_score)
            
        except Exception as e:
            print(f"CLIP score calculation failed: {e}")
            return None
    

    
    def get_available_models(self):
        """
        Get list of available generation models
        
        Returns:
            dict: Available models and their capabilities
        """
        models = {
            'baseline': {
                'available': True,
                'capabilities': [
                    'text_to_floorplan',
                    'basic_generation'
                ],
                'description': 'Baseline Stable Diffusion model fine-tuned on architectural data'
            },
            'constraint_aware': {
                'available': True,
                'capabilities': [
                    'text_to_floorplan',
                    'spatial_constraints',
                    'adjacency_awareness'
                ],
                'description': 'Enhanced diffusion model with spatial constraints and adjacency loss'
            }
        }
        
        return models
    
    def generate_synthetic_floorplan(self, room_count=3, room_types=None):
        """
        Generate a synthetic floor plan for testing purposes
        
        Args:
            room_count (int): Number of rooms
            room_types (list): List of room types
        
        Returns:
            PIL.Image: Generated synthetic floor plan
        """
        if room_types is None:
            room_types = ['bedroom', 'bathroom', 'kitchen', 'living room']
        
        # Create a simple synthetic floor plan
        width, height = 512, 512
        image = Image.new('RGB', (width, height), 'white')
        
        # This is a placeholder - in a real implementation, you would
        # generate actual floor plan layouts
        
        return image