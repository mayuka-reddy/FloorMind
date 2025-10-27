#!/usr/bin/env python3
"""
Simple FloorMind Training Script
A streamlined training script for fine-tuning Stable Diffusion on floor plans
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Diffusion model imports
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

class FloorPlanDataset(Dataset):
    """Dataset class for floor plan images and descriptions"""
    
    def __init__(self, metadata_file: str, image_dir: str, transform=None, max_samples=None):
        """Initialize dataset"""
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load metadata if available
        if Path(metadata_file).exists():
            self.metadata = pd.read_csv(metadata_file)
            if max_samples:
                self.metadata = self.metadata.head(max_samples)
        else:
            # Create dummy metadata
            image_files = list(self.image_dir.glob("*.png"))
            if max_samples:
                image_files = image_files[:max_samples]
            
            self.metadata = pd.DataFrame({
                'image_path': [str(f.relative_to(self.image_dir.parent)) for f in image_files],
                'description': [f"Floor plan layout {i}" for i in range(len(image_files))]
            })
        
        # Generate descriptions
        self.descriptions = self._generate_descriptions()
        
        print(f"📊 Dataset loaded: {len(self.metadata)} samples")
    
    def _generate_descriptions(self) -> List[str]:
        """Generate text descriptions for floor plans"""
        if 'description' in self.metadata.columns:
            return self.metadata['description'].tolist()
        
        base_descriptions = [
            "A detailed architectural floor plan",
            "Modern residential floor plan layout", 
            "Architectural blueprint of a house",
            "Floor plan with rooms and corridors",
            "Residential building floor layout",
            "Architectural drawing of apartment layout",
            "House floor plan with multiple rooms",
            "Building blueprint with room divisions"
        ]
        
        descriptions = []
        for i in range(len(self.metadata)):
            desc = base_descriptions[i % len(base_descriptions)]
            descriptions.append(desc)
        
        return descriptions
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get image path
        if 'image_path' in self.metadata.columns:
            image_path = Path(self.metadata.iloc[idx]['image_path'])
            if not image_path.is_absolute():
                image_path = self.image_dir.parent / image_path
        else:
            # Fallback to finding images
            image_files = list(self.image_dir.glob("*.png"))
            if idx < len(image_files):
                image_path = image_files[idx]
            else:
                image_path = image_files[0] if image_files else None
        
        # Load image
        try:
            if image_path and image_path.exists():
                image = Image.open(image_path).convert('RGB')
            else:
                # Create dummy image
                image = Image.new('RGB', (512, 512), color='white')
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new('RGB', (512, 512), color='white')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get description
        description = self.descriptions[idx]
        
        return {
            'image': image,
            'text': description,
            'idx': idx
        }

class SimpleFloorMindTrainer:
    """Simple trainer for FloorMind base model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🚀 Initializing FloorMind Trainer")
        print(f"📱 Device: {self.device}")
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Initialize components
        self._load_model_components()
        self._setup_dataset()
        self._setup_training()
        
    def _load_model_components(self):
        """Load Stable Diffusion model components"""
        print("🔄 Loading Stable Diffusion model components...")
        
        model_name = self.config["model_name"]
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        
        # Load UNet (this is what we'll fine-tune)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # Move to device
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)
        
        # Freeze VAE and text encoder (only train UNet)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Enable training mode for UNet
        self.unet.train()
        
        print("✅ Model components loaded successfully")
        print("🔒 VAE and text encoder frozen")
        print("🎯 UNet ready for training")
    
    def _setup_dataset(self):
        """Setup dataset and dataloader"""
        print("📊 Setting up dataset...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((self.config["resolution"], self.config["resolution"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Create dataset
        self.dataset = FloorPlanDataset(
            metadata_file=self.config["metadata_file"],
            image_dir=self.config["images_dir"],
            transform=transform,
            max_samples=self.config.get("max_samples")
        )
        
        # Create dataloader (no multiprocessing to avoid pickling issues)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        print(f"✅ Dataset ready: {len(self.dataset)} samples, {len(self.dataloader)} batches")
    
    def _setup_training(self):
        """Setup optimizer and scheduler"""
        print("⚙️ Setting up training components...")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-08
        )
        
        # Setup learning rate scheduler
        num_training_steps = len(self.dataloader) * self.config["num_epochs"]
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_training_steps,
            eta_min=self.config["learning_rate"] * 0.1
        )
        
        # Training metrics
        self.training_stats = {
            'epoch': [],
            'step': [],
            'loss': [],
            'lr': [],
            'timestamp': []
        }
        
        print(f"✅ Training setup complete")
        print(f"🔄 Total training steps: {num_training_steps}")
    
    def encode_text(self, text_batch):
        """Encode text prompts to embeddings"""
        text_inputs = self.tokenizer(
            text_batch,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def training_step(self, batch):
        """Single training step"""
        images = batch['image'].to(self.device)
        texts = batch['text']
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        text_embeddings = self.encode_text(texts)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting training...")
        print(f"📊 Training for {self.config['num_epochs']} epochs")
        
        global_step = 0
        start_time = datetime.now()
        
        for epoch in range(self.config["num_epochs"]):
            epoch_losses = []
            
            progress_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
                leave=True
            )
            
            for step, batch in enumerate(progress_bar):
                # Forward pass
                loss = self.training_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.config["max_grad_norm"])
                
                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Track metrics
                current_loss = loss.detach().item()
                epoch_losses.append(current_loss)
                
                # Log progress
                if global_step % self.config["logging_steps"] == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    
                    self.training_stats['epoch'].append(epoch)
                    self.training_stats['step'].append(global_step)
                    self.training_stats['loss'].append(current_loss)
                    self.training_stats['lr'].append(current_lr)
                    self.training_stats['timestamp'].append(datetime.now())
                    
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                global_step += 1
                
                # Save checkpoint
                if global_step % self.config["save_steps"] == 0:
                    self.save_checkpoint(global_step)
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   Steps: {len(epoch_losses)}")
            
            # Generate test image every few epochs
            if (epoch + 1) % 2 == 0:
                self.generate_test_image(epoch + 1)
        
        total_time = datetime.now() - start_time
        print(f"\n🎉 Training completed!")
        print(f"⏱️ Total time: {total_time}")
        print(f"📈 Total steps: {global_step}")
        
        # Save final model
        self.save_final_model()
        
        # Generate training statistics
        self.save_training_stats()
        
        return global_step
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config["output_dir"]) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save UNet state
        torch.save(self.unet.state_dict(), checkpoint_dir / "unet.pth")
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pth")
        torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pth")
        
        print(f"💾 Checkpoint saved at step {step}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_model_dir = Path(self.config["output_dir"]) / "final_model"
        final_model_dir.mkdir(exist_ok=True)
        
        # Save the fine-tuned UNet
        self.unet.save_pretrained(final_model_dir / "unet")
        
        # Save other components (unchanged but needed for pipeline)
        self.tokenizer.save_pretrained(final_model_dir / "tokenizer")
        self.text_encoder.save_pretrained(final_model_dir / "text_encoder")
        self.vae.save_pretrained(final_model_dir / "vae")
        self.noise_scheduler.save_pretrained(final_model_dir / "scheduler")
        
        # Save training configuration
        with open(final_model_dir / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        print(f"💾 Final model saved to: {final_model_dir}")
    
    def generate_test_image(self, epoch):
        """Generate test image during training"""
        print(f"🎨 Generating test image at epoch {epoch}...")
        
        # Create pipeline
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        
        # Generate image
        prompt = "A detailed architectural floor plan"
        
        with torch.no_grad():
            image = pipeline(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
        
        # Save image
        output_dir = Path(self.config["output_dir"])
        image.save(output_dir / f"test_epoch_{epoch}.png")
        
        print(f"🖼️ Test image saved for epoch {epoch}")
    
    def save_training_stats(self):
        """Save training statistics and visualizations"""
        if not self.training_stats['step']:
            print("⚠️ No training statistics to save")
            return
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(self.training_stats)
        
        # Save CSV
        output_dir = Path(self.config["output_dir"])
        stats_df.to_csv(output_dir / "training_stats.csv", index=False)
        
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(stats_df['step'], stats_df['loss'], alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate curve
        ax2.plot(stats_df['step'], stats_df['lr'], color='orange', alpha=0.7)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training statistics saved")
        print(f"   Final Loss: {stats_df['loss'].iloc[-1]:.4f}")
        print(f"   Average Loss: {stats_df['loss'].mean():.4f}")

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        # Data paths
        "metadata_file": "../data/metadata.csv",
        "images_dir": "../data/processed/images",
        
        # Model configuration
        "model_name": "runwayml/stable-diffusion-v1-5",
        "resolution": 512,
        "train_batch_size": 2,  # Reduced for stability
        
        # Training parameters
        "num_epochs": 5,  # Reduced for quick training
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,
        
        # Output configuration
        "output_dir": "../outputs/models/base_model",
        "save_steps": 100,
        "logging_steps": 10,
        
        # Quick test mode
        "max_samples": 50,  # Limit samples for quick testing
    }
    
    print("🚀 FloorMind Simple Training")
    print("=" * 40)
    
    try:
        # Initialize trainer
        trainer = SimpleFloorMindTrainer(config)
        
        # Start training
        total_steps = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Total steps: {total_steps}")
        print(f"📁 Model saved to: {config['output_dir']}")
        
        print(f"\n💡 Next steps:")
        print("1. Check generated test images")
        print("2. Run inference with the trained model")
        print("3. Fine-tune with constraint-aware training")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)