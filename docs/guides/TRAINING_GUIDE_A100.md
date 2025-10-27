# FloorMind Training Guide - Google Colab A100 Optimized

Complete guide for training FloorMind on Google Colab with A100 GPU, including hyperparameter optimization, metrics tracking, and best practices.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup & Configuration](#setup--configuration)
3. [Hyperparameters Explained](#hyperparameters-explained)
4. [Metrics Tracking](#metrics-tracking)
5. [Memory Optimization](#memory-optimization)
6. [Training Process](#training-process)
7. [Model Evaluation](#model-evaluation)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements
- **GPU**: Google Colab A100 (40GB or 80GB VRAM)
- **RAM**: 12GB+ system RAM
- **Storage**: 50GB+ free space for dataset and outputs

### Software Requirements
```bash
Python 3.8+
PyTorch 2.0+
CUDA 11.8+
diffusers==0.35.2
transformers==4.57.1
accelerate==1.10.1
```

### Dataset
- **CubiCasa5K** processed as numpy arrays
- Files needed:
  - `train_images.npy` (N, 512, 512, 3)
  - `train_descriptions.npy` (N,)
  - `test_images.npy` (M, 512, 512, 3)
  - `test_descriptions.npy` (M,)

---

## Setup & Configuration

### 1. Open Google Colab
```python
# Check GPU allocation
!nvidia-smi

# Expected output: Tesla A100-SXM4-40GB or A100-SXM4-80GB
```

### 2. Load Configuration
```python
import json

# Load optimized config
with open('config/training_config_a100.json', 'r') as f:
    config = json.load(f)

# Adjust based on your GPU memory
if gpu_memory_gb < 40:
    config['training_hyperparameters']['train_batch_size'] = 1
    config['training_hyperparameters']['gradient_accumulation_steps'] = 8
```

### 3. Upload Dataset
```python
from google.colab import files

# Upload your preprocessed numpy files
uploaded = files.upload()

# Files will be in /content/
```

---

## Hyperparameters Explained

### Core Training Parameters

#### Learning Rate: `5e-6`
- **Why**: Lower LR ensures stable training for fine-tuning
- **Range**: 1e-6 to 1e-5
- **Impact**: Higher = faster convergence but risk of instability

#### Batch Size: `2` (Effective: `8`)
- **Actual batch size**: 2 samples per step
- **Gradient accumulation**: 4 steps
- **Effective batch size**: 2 × 4 = 8
- **Why**: Balances memory usage and training stability

#### Epochs: `15`
- **Why**: Sufficient for convergence on CubiCasa5K
- **Typical range**: 10-20 epochs
- **Monitor**: Stop if validation loss plateaus

#### Weight Decay: `0.01`
- **Purpose**: Regularization to prevent overfitting
- **Range**: 0.001 to 0.1
- **Impact**: Higher = more regularization

### Diffusion Parameters

#### Noise Schedule: `scaled_linear`
- **Why**: Better for architectural images than linear
- **Alternatives**: `linear`, `cosine`, `squaredcos_cap_v2`
- **Impact**: Affects noise distribution during training

#### Timesteps: `1000`
- **Standard**: 1000 steps for Stable Diffusion
- **Range**: 500-1000
- **Impact**: More steps = finer control but slower

### Optimizer Settings

#### AdamW 8-bit
- **Memory savings**: ~50% compared to standard AdamW
- **Performance**: Minimal impact on convergence
- **Betas**: (0.9, 0.999) - standard for diffusion models
- **Epsilon**: 1e-8 - numerical stability

---

## Metrics Tracking

### Primary Metrics

#### 1. Training Loss
```python
# MSE between predicted and actual noise
loss = F.mse_loss(noise_pred, noise_target)
```
- **Target**: < 0.1 for good convergence
- **Monitor**: Should decrease steadily
- **Warning**: If increasing, reduce learning rate

#### 2. Validation Loss
```python
# Computed every 250 steps
val_loss = validate_model(val_dataloader)
```
- **Target**: Close to training loss (< 0.15)
- **Monitor**: Gap indicates overfitting
- **Action**: If gap > 0.05, increase regularization

#### 3. Learning Rate
```python
# Cosine schedule with warm restarts
current_lr = scheduler.get_last_lr()[0]
```
- **Pattern**: Decreases then restarts
- **Monitor**: Ensure it's decreasing overall
- **Range**: 5e-6 to 5e-7 by end

#### 4. GPU Memory Usage
```python
gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
```
- **Target**: < 38GB for A100-40GB
- **Monitor**: Spikes indicate memory leaks
- **Action**: If > 90%, reduce batch size

### Secondary Metrics

#### Gradient Norm
```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0
)
```
- **Target**: 0.1 - 1.0
- **Warning**: > 10 indicates instability
- **Action**: Reduce learning rate if consistently high

#### Training Speed
- **Target**: 2-3 seconds per step
- **Monitor**: Slowdowns indicate bottlenecks
- **Factors**: I/O, data loading, GPU utilization

### Metrics Visualization

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load metrics
df = pd.read_csv('outputs/metrics/training_metrics.csv')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(df['step'], df['train_loss'], label='Train')
axes[0, 0].plot(df['step'], df['val_loss'], label='Val')
axes[0, 0].set_title('Loss Over Time')
axes[0, 0].legend()

# Learning Rate
axes[0, 1].plot(df['step'], df['learning_rate'])
axes[0, 1].set_title('Learning Rate Schedule')

# GPU Memory
axes[1, 0].plot(df['step'], df['gpu_memory_gb'])
axes[1, 0].set_title('GPU Memory Usage (GB)')

# Gradient Norm
axes[1, 1].plot(df['step'], df['gradient_norm'])
axes[1, 1].set_title('Gradient Norm')

plt.tight_layout()
plt.savefig('outputs/metrics/training_curves.png', dpi=300)
```

---

## Memory Optimization

### Techniques Used

#### 1. Mixed Precision (FP16)
```python
accelerator = Accelerator(mixed_precision="fp16")
```
- **Memory savings**: ~50%
- **Speed improvement**: 2-3x faster
- **Trade-off**: Minimal accuracy loss

#### 2. Gradient Checkpointing
```python
unet.enable_gradient_checkpointing()
```
- **Memory savings**: ~30%
- **Speed impact**: ~20% slower
- **Worth it**: Yes, for larger batch sizes

#### 3. XFormers Attention
```python
unet.enable_xformers_memory_efficient_attention()
```
- **Memory savings**: ~20%
- **Speed improvement**: 10-15% faster
- **Requirement**: xformers library installed

#### 4. 8-bit Optimizer
```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(params, lr=5e-6)
```
- **Memory savings**: ~50% optimizer states
- **Performance**: Negligible impact
- **Compatibility**: Works with A100

#### 5. Gradient Accumulation
```python
# Effective batch size = 2 × 4 = 8
gradient_accumulation_steps = 4
```
- **Memory savings**: Train with smaller batches
- **Benefit**: Larger effective batch size
- **Trade-off**: Slower training (4x steps)

### Memory Budget (A100-40GB)

```
Component                Memory Usage
─────────────────────────────────────
Model (UNet)            ~3.5 GB
VAE                     ~0.3 GB
Text Encoder            ~0.5 GB
Optimizer States        ~7.0 GB (8-bit)
Activations             ~8.0 GB
Batch Data              ~2.0 GB
Gradients               ~3.5 GB
Buffer                  ~5.0 GB
─────────────────────────────────────
Total                   ~29.8 GB
Available               ~10.2 GB (buffer)
```

---

## Training Process

### Step-by-Step Execution

#### 1. Initialize Training
```python
# Start training
print("🚀 Starting training...")
global_step = 0
best_val_loss = float('inf')
start_time = datetime.now()
```

#### 2. Training Loop
```python
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # Forward pass
        loss = training_step(batch)
        
        # Backward pass
        accelerator.backward(loss)
        
        # Optimizer step (every N accumulation steps)
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Logging
        if global_step % logging_steps == 0:
            log_metrics(loss, lr, gpu_memory)
        
        # Validation
        if global_step % eval_steps == 0:
            val_loss = validate()
            save_checkpoint_if_best(val_loss)
        
        # Checkpointing
        if global_step % save_steps == 0:
            save_checkpoint(global_step)
```

#### 3. Validation
```python
def validate_model():
    unet.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            loss = compute_loss(batch)
            val_losses.append(loss.item())
    
    unet.train()
    return np.mean(val_losses)
```

#### 4. Checkpointing
```python
# Save checkpoint
checkpoint_dir = f"outputs/checkpoints/step-{global_step}"
accelerator.save_state(checkpoint_dir)

# Save best model
if val_loss < best_val_loss:
    best_val_loss = val_loss
    accelerator.save_state("outputs/models/best_model")
```

### Expected Training Time

| Dataset Size | Epochs | A100-40GB | A100-80GB |
|--------------|--------|-----------|-----------|
| 1,000 images | 15     | ~1 hour   | ~45 min   |
| 5,000 images | 15     | ~4 hours  | ~3 hours  |
| 10,000 images| 15     | ~8 hours  | ~6 hours  |

---

## Model Evaluation

### Quantitative Metrics

#### 1. Final Loss Values
```python
# Target values
train_loss_final < 0.08  # Excellent
val_loss_final < 0.12    # Good generalization
loss_gap < 0.05          # No overfitting
```

#### 2. Generation Quality
```python
# Generate test images
test_prompts = [
    "Modern 3-bedroom apartment floor plan",
    "Traditional house with garage",
    "Open concept loft design"
]

for prompt in test_prompts:
    image = pipeline(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    # Visual inspection
    image.save(f"test_{prompt[:20]}.png")
```

### Qualitative Assessment

#### Visual Quality Checklist
- [ ] Clear room boundaries
- [ ] Proper wall thickness
- [ ] Realistic proportions
- [ ] Correct room labels
- [ ] Architectural accuracy
- [ ] No artifacts or noise

#### Architectural Accuracy
- [ ] Doors in logical positions
- [ ] Windows on exterior walls
- [ ] Proper circulation paths
- [ ] Functional room layouts
- [ ] Building code compliance

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
**Symptoms**: CUDA out of memory error

**Solutions**:
```python
# Reduce batch size
config['train_batch_size'] = 1

# Increase gradient accumulation
config['gradient_accumulation_steps'] = 8

# Enable all optimizations
config['gradient_checkpointing'] = True
config['use_8bit_adam'] = True
config['enable_xformers'] = True
```

#### 2. Loss Not Decreasing
**Symptoms**: Loss stays constant or increases

**Solutions**:
```python
# Reduce learning rate
config['learning_rate'] = 1e-6

# Check data loading
print(f"Batch shape: {batch['image'].shape}")
print(f"Value range: [{batch['image'].min()}, {batch['image'].max()}]")

# Verify model is training
print(f"UNet training: {unet.training}")
```

#### 3. Slow Training
**Symptoms**: < 1 step per second

**Solutions**:
```python
# Reduce num_workers if I/O bound
config['dataloader_num_workers'] = 0

# Enable XFormers
unet.enable_xformers_memory_efficient_attention()

# Check GPU utilization
!nvidia-smi
```

#### 4. Poor Generation Quality
**Symptoms**: Blurry or unrealistic outputs

**Solutions**:
```python
# Train longer
config['num_epochs'] = 20

# Adjust guidance scale
guidance_scale = 9.0  # Higher = more prompt adherence

# More inference steps
num_inference_steps = 50
```

### Debug Commands

```python
# Check model status
print(f"Model device: {next(unet.parameters()).device}")
print(f"Model dtype: {next(unet.parameters()).dtype}")
print(f"Trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

# Monitor GPU
import subprocess
subprocess.run(['nvidia-smi', 'dmon', '-s', 'u'])

# Check data pipeline
sample = next(iter(train_dataloader))
print(f"Image shape: {sample['image'].shape}")
print(f"Text sample: {sample['text'][0]}")
```

---

## Best Practices

### 1. Start Small
- Test with 100 images first
- Verify pipeline works end-to-end
- Then scale to full dataset

### 2. Monitor Continuously
- Check metrics every 50 steps
- Validate every 250 steps
- Save checkpoints every 500 steps

### 3. Save Everything
- Training configuration
- All checkpoints
- Metrics CSV
- Generated test images
- Training logs

### 4. Experiment Systematically
- Change one parameter at a time
- Document all experiments
- Compare metrics objectively

### 5. Use Version Control
```bash
git add config/training_config_a100.json
git commit -m "Update training config for experiment X"
```

---

## Output Files

After training, you'll have:

```
outputs/
├── models/
│   ├── best_model/
│   │   ├── unet/
│   │   ├── vae/
│   │   ├── text_encoder/
│   │   ├── tokenizer/
│   │   └── scheduler/
│   └── floormind_model.pkl
├── metrics/
│   ├── training_metrics.csv
│   ├── training_summary.json
│   └── training_curves.png
├── checkpoints/
│   ├── step-500/
│   ├── step-1000/
│   └── step-1500/
└── test_generations/
    ├── test_01.png
    ├── test_02.png
    └── test_03.png
```

---

## Next Steps

1. **Evaluate Model**: Generate test images and assess quality
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy**: Integrate with backend API
4. **Iterate**: Continue training or try different architectures

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review training logs in `outputs/logs/`
- Consult [Diffusers Documentation](https://huggingface.co/docs/diffusers)

---

**Happy Training! 🚀**