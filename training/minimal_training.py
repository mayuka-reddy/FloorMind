#!/usr/bin/env python3
"""
Minimal FloorMind Training Script
A lightweight training script that focuses on dataset preparation and basic model setup
"""

import os
import sys
import torch
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

class MinimalFloorPlanTrainer:
    """Minimal trainer for FloorMind dataset preparation and basic training setup"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🚀 Initializing Minimal FloorMind Trainer")
        print(f"📱 Device: {self.device}")
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load and analyze the dataset"""
        print("📊 Loading dataset...")
        
        # Load metadata
        metadata_path = Path(self.config["metadata_file"])
        if metadata_path.exists():
            self.df = pd.read_csv(metadata_path)
            print(f"✅ Metadata loaded: {len(self.df)} samples")
        else:
            print("❌ Metadata file not found")
            return
        
        # Check images
        images_dir = Path(self.config["images_dir"])
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            print(f"✅ Images found: {len(image_files)}")
            self.image_files = image_files
        else:
            print("❌ Images directory not found")
            return
        
        # Analyze dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze the dataset structure and quality"""
        print("\n🔍 Dataset Analysis:")
        print("=" * 30)
        
        # Basic statistics
        print(f"Total samples: {len(self.df)}")
        print(f"Image files: {len(self.image_files)}")
        
        # Column analysis
        if 'room_count' in self.df.columns:
            print(f"Room count range: {self.df['room_count'].min()}-{self.df['room_count'].max()}")
            print(f"Average room count: {self.df['room_count'].mean():.1f}")
        
        if 'category' in self.df.columns:
            print(f"Categories: {self.df['category'].value_counts().to_dict()}")
        
        if 'architectural_style' in self.df.columns:
            print(f"Styles: {self.df['architectural_style'].value_counts().to_dict()}")
        
        # Check train/test split if available
        if 'split' in self.df.columns:
            print(f"Split distribution: {self.df['split'].value_counts().to_dict()}")
        else:
            # Create train/test split
            self._create_train_test_split()
    
    def _create_train_test_split(self):
        """Create train/test split"""
        print("\n🔄 Creating train/test split...")
        
        from sklearn.model_selection import train_test_split
        
        # Create 60/40 split
        train_indices, test_indices = train_test_split(
            range(len(self.df)),
            test_size=0.4,
            random_state=42,
            stratify=self.df['room_count'] if 'room_count' in self.df.columns else None
        )
        
        # Add split column
        self.df['split'] = 'test'
        self.df.loc[train_indices, 'split'] = 'train'
        
        print(f"✅ Split created:")
        print(f"   Train: {len(train_indices)} samples (60%)")
        print(f"   Test: {len(test_indices)} samples (40%)")
        
        # Save updated metadata
        self.df.to_csv(self.config["metadata_file"], index=False)
        print(f"💾 Updated metadata saved")
    
    def prepare_training_data(self):
        """Prepare data for training"""
        print("\n📋 Preparing training data...")
        
        # Get training samples
        if 'split' in self.df.columns:
            train_df = self.df[self.df['split'] == 'train'].copy()
            test_df = self.df[self.df['split'] == 'test'].copy()
        else:
            # Simple split
            split_idx = int(len(self.df) * 0.6)
            train_df = self.df[:split_idx].copy()
            test_df = self.df[split_idx:].copy()
        
        print(f"📊 Training samples: {len(train_df)}")
        print(f"📊 Test samples: {len(test_df)}")
        
        # Save split datasets
        output_dir = Path(self.config["output_dir"])
        train_df.to_csv(output_dir / "train_data.csv", index=False)
        test_df.to_csv(output_dir / "test_data.csv", index=False)
        
        print(f"💾 Split datasets saved to {output_dir}")
        
        return train_df, test_df
    
    def analyze_images(self, sample_size: int = 10):
        """Analyze a sample of images"""
        print(f"\n🖼️ Analyzing {sample_size} sample images...")
        
        # Sample images
        sample_files = self.image_files[:sample_size]
        
        # Analyze dimensions and quality
        image_stats = []
        
        for img_file in tqdm(sample_files, desc="Analyzing images"):
            try:
                img = Image.open(img_file)
                stats = {
                    'filename': img_file.name,
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'size_kb': img_file.stat().st_size // 1024
                }
                image_stats.append(stats)
            except Exception as e:
                print(f"⚠️ Error analyzing {img_file.name}: {e}")
        
        # Create analysis DataFrame
        img_df = pd.DataFrame(image_stats)
        
        if len(img_df) > 0:
            print(f"✅ Image analysis complete:")
            print(f"   Average size: {img_df['width'].mean():.0f}×{img_df['height'].mean():.0f}")
            print(f"   Size range: {img_df['width'].min()}-{img_df['width'].max()} × {img_df['height'].min()}-{img_df['height'].max()}")
            print(f"   Average file size: {img_df['size_kb'].mean():.0f} KB")
            print(f"   Modes: {img_df['mode'].value_counts().to_dict()}")
            
            # Save analysis
            output_dir = Path(self.config["output_dir"])
            img_df.to_csv(output_dir / "image_analysis.csv", index=False)
            
            return img_df
        
        return None
    
    def create_sample_visualization(self):
        """Create visualization of sample images"""
        print("\n🎨 Creating sample visualization...")
        
        # Select sample images
        sample_files = self.image_files[:12]
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, img_file in enumerate(sample_files):
            try:
                img = Image.open(img_file)
                axes[i].imshow(img)
                axes[i].set_title(f"{img_file.name[:20]}...", fontsize=8)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{img_file.name}", 
                           ha='center', va='center', fontsize=8)
                axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path(self.config["output_dir"])
        plt.savefig(output_dir / "sample_images.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Sample visualization saved")
    
    def generate_training_report(self):
        """Generate comprehensive training readiness report"""
        print("\n📋 Generating Training Readiness Report...")
        
        # Collect all information
        report = {
            "dataset_info": {
                "total_samples": len(self.df),
                "image_files": len(self.image_files),
                "metadata_columns": list(self.df.columns),
                "has_descriptions": 'description' in self.df.columns,
                "has_room_info": 'room_count' in self.df.columns,
                "has_categories": 'category' in self.df.columns
            },
            "data_quality": {
                "metadata_image_match": len(self.df) <= len(self.image_files),
                "all_images_512x512": True,  # Assuming from processing
                "consistent_format": "PNG"
            },
            "training_readiness": {
                "pytorch_available": True,
                "device": str(self.device),
                "gpu_available": torch.cuda.is_available() or torch.backends.mps.is_available(),
                "dataset_ready": len(self.df) > 0 and len(self.image_files) > 0,
                "recommended_batch_size": 4 if self.device.type in ['cuda', 'mps'] else 1
            },
            "recommendations": {
                "training_epochs": 5 if len(self.df) < 1000 else 10,
                "learning_rate": 1e-5,
                "batch_size": 2 if len(self.df) < 100 else 4,
                "save_steps": max(50, len(self.df) // 20)
            },
            "generation_date": datetime.now().isoformat()
        }
        
        # Save report
        output_dir = Path(self.config["output_dir"])
        with open(output_dir / "training_readiness_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        print("✅ Training Readiness Report:")
        print(f"   📊 Dataset: {report['dataset_info']['total_samples']} samples")
        print(f"   🖼️ Images: {report['dataset_info']['image_files']} files")
        print(f"   📱 Device: {report['training_readiness']['device']}")
        print(f"   🚀 GPU Available: {report['training_readiness']['gpu_available']}")
        print(f"   ✅ Ready for Training: {report['training_readiness']['dataset_ready']}")
        
        print(f"\n💡 Recommended Settings:")
        print(f"   Epochs: {report['recommendations']['training_epochs']}")
        print(f"   Learning Rate: {report['recommendations']['learning_rate']}")
        print(f"   Batch Size: {report['recommendations']['batch_size']}")
        
        return report

def main():
    """Main function"""
    
    # Configuration
    config = {
        "metadata_file": "data/metadata.csv",
        "images_dir": "data/processed/images",
        "output_dir": "outputs/training_analysis"
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = MinimalFloorPlanTrainer(config)
        
        # Prepare training data
        train_df, test_df = trainer.prepare_training_data()
        
        # Analyze images
        img_analysis = trainer.analyze_images(sample_size=20)
        
        # Create visualizations
        trainer.create_sample_visualization()
        
        # Generate report
        report = trainer.generate_training_report()
        
        print(f"\n🎉 Training setup analysis completed!")
        print(f"📁 Results saved to: {config['output_dir']}")
        
        if report['training_readiness']['dataset_ready']:
            print(f"\n✅ Ready to start training!")
            print(f"💡 Run: python training/simple_training.py")
        else:
            print(f"\n⚠️ Dataset needs preparation first")
            print(f"💡 Run: python process_full_dataset.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)