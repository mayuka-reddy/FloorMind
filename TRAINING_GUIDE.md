# FloorMind Training Guide

Complete guide for training FloorMind's AI models using interactive Jupyter notebooks.

## Overview

FloorMind uses a two-stage notebook-based training approach:
1. **Base Model Training**: `notebooks/FloorMind_Base_Training.ipynb`
2. **Constraint-Aware Fine-Tuning**: `notebooks/FloorMind_Constraint_FineTuning.ipynb`

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended) or Apple Silicon Mac
- 16GB+ RAM
- 50GB+ free disk space

### Dependencies
All dependencies are automatically installed within the notebooks:
```bash
# PyTorch (compatible version)
pip install torch==2.6.0 torchvision torchaudio

# Diffusion models
pip install diffusers transformers accelerate

# Data processing
pip install pandas numpy matplotlib seaborn scikit-learn
pip install pillow tqdm jupyter ipykernel opencv-python
```

## Training Notebooks

### 📓 Base Model Training Notebook

**File**: `notebooks/FloorMind_Base_Training.ipynb`

**What it does**:
- Complete environment setup and package installation
- Dataset loading and analysis with visualizations
- Base Stable Diffusion fine-tuning on floor plans
- Real-time training monitoring and statistics
- Model evaluation and test generation
- Comprehensive training analysis

**Key Features**:
- ✅ Automatic PyTorch installation and verification
- ✅ Interactive dataset exploration
- ✅ Live training progress with loss curves
- ✅ Automatic model saving and checkpointing
- ✅ Test image generation and quality assessment
- ✅ Complete training statistics and reports

**Usage**:
1. Open Jupyter: `jupyter notebook`
2. Navigate to `notebooks/FloorMind_Base_Training.ipynb`
3. Run all cells sequentially
4. Monitor training progress in real-time

### 📓 Constraint-Aware Fine-Tuning Notebook

**File**: `notebooks/FloorMind_Constraint_FineTuning.ipynb`

**What it does**:
- Load pre-trained base model or start from Stable Diffusion
- Implement architectural constraint loss functions
- Fine-tune with connectivity and structural constraints
- Generate constraint-aware floor plans
- Compare base vs constraint-aware models
- Comprehensive evaluation and analysis

**Key Features**:
- ✅ Advanced constraint loss functions
- ✅ Architectural rule enforcement
- ✅ Connectivity and circulation optimization
- ✅ Multi-component loss tracking
- ✅ Model comparison and evaluation
- ✅ Constraint-aware generation testing

**Usage**:
1. Complete base training first (or have a pre-trained model)
2. Open `notebooks/FloorMind_Constraint_FineTuning.ipynb`
3. Run all cells to fine-tune with constraints
4. Compare results with base model

## Training Process

### Stage 1: Base Model Training

```bash
# Start Jupyter
jupyter notebook

# Open and run: notebooks/FloorMind_Base_Training.ipynb
```

**Training Configuration** (embedded in notebook):
- Epochs: 10
- Learning Rate: 1e-5
- Batch Size: 4
- Resolution: 512x512
- Mixed Precision: FP16

**Outputs**:
- Model: `outputs/models/base_model/final_model/`
- Statistics: `training_stats.csv`
- Visualizations: `training_curves.png`
- Test Images: `test_generation_*.png`

### Stage 2: Constraint-Aware Fine-Tuning

```bash
# Open and run: notebooks/FloorMind_Constraint_FineTuning.ipynb
```

**Fine-Tuning Configuration**:
- Epochs: 5 (fewer for fine-tuning)
- Learning Rate: 5e-6 (lower for stability)
- Batch Size: 2
- Constraint Weights: Architectural (0.05), Connectivity (0.03)

**Outputs**:
- Model: `outputs/models/constraint_model/final_model/`
- Statistics: `finetuning_stats.csv`
- Analysis: `finetuning_analysis.png`
- Constraint Images: `constraint_test_*.png`
## Datase
t Preparation

### Automatic Dataset Setup
Both notebooks include automatic dataset handling:

1. **Check existing data**: Automatically detects processed datasets
2. **Load metadata**: Reads CSV files with image information
3. **Fallback options**: Creates synthetic data if needed
4. **Visual analysis**: Shows sample images and statistics

### Custom Dataset Integration
To use your own dataset:

1. Place images in: `data/processed/images/`
2. Create metadata CSV: `data/metadata.csv`
3. Update paths in notebook configuration cells

## Interactive Features

### 📊 Real-Time Monitoring
- Live loss curves and training progress
- Learning rate schedules
- Memory usage tracking
- Step-by-step execution with immediate feedback

### 🎨 Visual Analysis
- Dataset sample visualization
- Training curve plotting
- Generated image galleries
- Model comparison displays

### 📈 Comprehensive Statistics
- Training metrics tracking
- Loss component analysis
- Model performance evaluation
- Improvement calculations

## Notebook Advantages

### ✅ Complete Environment Control
- Automatic dependency installation
- Version compatibility handling
- Environment verification

### ✅ Interactive Development
- Step-by-step execution
- Immediate results visualization
- Easy parameter adjustment
- Real-time debugging

### ✅ Comprehensive Documentation
- Inline explanations and markdown
- Code comments and examples
- Visual guides and tutorials
- Complete workflow documentation

### ✅ Reproducible Results
- Fixed random seeds
- Saved configurations
- Complete execution logs
- Shareable notebooks

## Troubleshooting

### Common Issues

**PyTorch Installation**:
- Notebooks automatically handle PyTorch installation
- Compatible versions are specified
- Verification steps included

**Memory Issues**:
- Reduce batch size in configuration cells
- Enable gradient accumulation
- Use mixed precision (FP16)

**Dataset Problems**:
- Notebooks include fallback synthetic data
- Automatic dataset validation
- Visual inspection tools

### Performance Optimization

**GPU Utilization**:
```python
# In notebook configuration cells
config = {
    "mixed_precision": "fp16",
    "gradient_accumulation_steps": 4,
    "dataloader_num_workers": 2
}
```

**Memory Management**:
```python
# Adjust in notebook
config = {
    "train_batch_size": 2,  # Reduce if needed
    "eval_batch_size": 1,
    "max_samples": 100      # For quick testing
}
```

## Model Outputs

### Base Model Results
- **Location**: `outputs/models/base_model/final_model/`
- **Components**: UNet, VAE, Text Encoder, Tokenizer, Scheduler
- **Statistics**: Complete training metrics and visualizations
- **Test Images**: Generated floor plan samples

### Constraint-Aware Model Results
- **Location**: `outputs/models/constraint_model/final_model/`
- **Enhanced Features**: Architectural constraints, connectivity rules
- **Comparison**: Side-by-side with base model
- **Evaluation**: Comprehensive constraint compliance analysis

## Integration with FloorMind

### Backend Integration
After training, update the model service:

```python
# In backend/services/model_service.py
MODEL_PATH = "outputs/models/constraint_model/final_model"
```

### API Usage
```python
from backend.services.model_service import ModelService

service = ModelService()
result = service.generate_floorplan(
    prompt="Modern 3-bedroom apartment with open concept",
    constraints={"connectivity": True, "structural": True}
)
```

## Expected Results

### Performance Metrics
- **Base Model Training**: ~2-4 hours on modern GPU
- **Fine-Tuning**: ~1-2 hours additional
- **Final Loss**: <0.1 (base), <0.08 (constraint-aware)
- **Generation Time**: 2-5 seconds per image

### Quality Improvements
- **Architectural Accuracy**: >85%
- **Constraint Compliance**: >90%
- **Visual Quality**: Professional-grade floor plans
- **Diversity**: High variation in generated layouts

## Next Steps

After completing notebook training:

1. **✅ Model Deployment**: Integrate trained models with FloorMind backend
2. **🧪 Testing**: Validate with real architectural requirements
3. **🔧 Optimization**: Fine-tune based on user feedback
4. **📈 Scaling**: Prepare for production deployment

## Support

### Notebook-Specific Help
- Each notebook includes troubleshooting sections
- Inline documentation and examples
- Visual debugging tools
- Automatic error handling

### Getting Help
- Check notebook output cells for detailed error messages
- Review configuration sections for parameter adjustments
- Use built-in visualization tools for debugging
- Refer to inline documentation and comments

---

*Interactive training made simple with Jupyter notebooks! 🚀📓*