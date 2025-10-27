# 🚀 Google Colab Training Guide - Step by Step

This guide will walk you through training FloorMind on Google Colab and integrating the model with your frontend.

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Google account (for Colab)
- ✅ Google Drive with at least 10GB free space
- ✅ CubiCasa5K dataset processed as numpy arrays
- ✅ Basic understanding of Jupyter notebooks

---

## 📁 Step 1: Prepare Your Dataset

### Required Files

You need these 4 files from your processed dataset:

```
📦 Dataset Files (Upload to Google Drive)
├── train_images.npy          # Training images (N, 512, 512, 3)
├── train_descriptions.npy     # Training descriptions (N,)
├── test_images.npy           # Test images (M, 512, 512, 3)
└── test_descriptions.npy      # Test descriptions (M,)
```

### Where to Get These Files

If you don't have these files yet, run the preprocessing notebook first:

1. Open `notebooks/CubiCasa5K_Processing_Enhanced.ipynb` locally
2. Process your CubiCasa5K dataset
3. It will generate the 4 numpy files above
4. Upload them to Google Drive

### Upload to Google Drive

```
📁 Google Drive Structure
└── FloorMind/
    └── data/
        ├── train_images.npy
        ├── train_descriptions.npy
        ├── test_images.npy
        └── test_descriptions.npy
```

**How to upload:**
1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `FloorMind/data/`
3. Upload all 4 `.npy` files to this folder

---

## 🎯 Step 2: Open the Training Notebook

### Which Notebook to Use

**Use this notebook:** `notebooks/FloorMind_Colab_Training.ipynb`

This is the **CORRECT** notebook that:
- ✅ Has all required code
- ✅ Generates model files for frontend integration
- ✅ Optimized for Google Colab A100
- ✅ Includes comprehensive metrics tracking
- ✅ Saves complete model pipeline

### How to Open in Colab

**Option 1: Direct Upload**
```bash
1. Go to https://colab.research.google.com/
2. Click "File" > "Upload notebook"
3. Select: notebooks/FloorMind_Colab_Training.ipynb
```

**Option 2: From GitHub** (if you've pushed to GitHub)
```bash
1. Go to https://colab.research.google.com/
2. Click "GitHub" tab
3. Enter your repository URL
4. Select: notebooks/FloorMind_Colab_Training.ipynb
```

---

## ⚙️ Step 3: Configure Colab Runtime

### Select A100 GPU

**IMPORTANT:** You MUST use A100 GPU for optimal training

```bash
1. In Colab, click "Runtime" menu
2. Select "Change runtime type"
3. Hardware accelerator: GPU
4. GPU type: A100 (if available) or T4
5. Click "Save"
```

### Verify GPU

Run this in the first cell:
```python
!nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla A100-SXM... Off  | 00000000:00:04.0 Off |                    0 |
| N/A   34C    P0    44W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## 📝 Step 4: Run the Training Notebook

### Cell-by-Cell Execution

Execute cells in order. Here's what each section does:

#### **Section 1: Environment Setup** (Cells 1-3)
```python
# Cell 1: Check Colab environment
# Cell 2: Install dependencies (takes ~3 minutes)
# Cell 3: Import libraries and verify GPU
```

**Expected output:**
```
✅ PyTorch: 2.0.1
✅ CUDA Available: True
🎮 GPU: Tesla A100-SXM4-40GB
💾 GPU Memory: 40.0 GB
```

#### **Section 2: Mount Google Drive** (Cell 4)
```python
# This will prompt you to authorize Google Drive access
```

**Steps:**
1. Click the link that appears
2. Sign in to your Google account
3. Copy the authorization code
4. Paste it back in Colab
5. Press Enter

#### **Section 3: Load Dataset** (Cells 5-6)
```python
# Cell 5: Set data paths
# Cell 6: Load numpy arrays from Google Drive
```

**Expected output:**
```
✅ Training images: (4500, 512, 512, 3)
✅ Training descriptions: 4500
✅ Test images: (500, 512, 512, 3)
✅ Test descriptions: 500
```

#### **Section 4: Configuration** (Cell 7)
```python
# Loads optimized hyperparameters for A100
```

**Key settings:**
- Batch size: 2
- Gradient accumulation: 4 (effective batch = 8)
- Epochs: 15
- Learning rate: 5e-6
- Mixed precision: FP16

#### **Section 5: Create Datasets** (Cell 8)
```python
# Creates PyTorch datasets and dataloaders
```

#### **Section 6: Load Model** (Cell 9)
```python
# Loads Stable Diffusion components
# This takes ~5 minutes
```

#### **Section 7: Setup Training** (Cell 10)
```python
# Configures optimizer and scheduler
```

#### **Section 8: Training Loop** (Cell 11)
```python
# THIS IS THE MAIN TRAINING CELL
# Takes ~4 hours for 5000 images
```

**What happens during training:**
- Progress bar shows current epoch and step
- Loss values displayed every 50 steps
- Validation runs every 250 steps
- Checkpoints saved every 500 steps
- GPU memory monitored continuously

**Expected output:**
```
🚀 Starting training...
📊 Training for 15 epochs

Epoch 1/15: 100%|██████████| 2250/2250 [15:23<00:00, loss=0.1234, lr=4.5e-06, gpu=35.2GB]
📊 Step 250 - Val Loss: 0.1156
💾 New best model saved! Val Loss: 0.1156

Epoch 2/15: 100%|██████████| 2250/2250 [15:20<00:00, loss=0.0987, lr=4.2e-06, gpu=35.1GB]
...
```

#### **Section 9: Save Model** (Cell 12)
```python
# Saves complete model for frontend integration
```

**Files generated:**
```
/content/outputs/models/floormind_baseline/
├── final_model/
│   ├── unet/                    # Fine-tuned UNet
│   ├── vae/                     # VAE encoder/decoder
│   ├── text_encoder/            # CLIP text encoder
│   ├── tokenizer/               # CLIP tokenizer
│   ├── scheduler/               # Noise scheduler
│   ├── training_config.json     # Training configuration
│   └── training_stats.csv       # Metrics history
├── floormind_pipeline/          # Complete pipeline
└── floormind_model.pkl          # Pickle file (for easy loading)
```

#### **Section 10: Test Generation** (Cell 13)
```python
# Generates test images to verify model works
```

#### **Section 11: Download Results** (Cell 14)
```python
# Creates zip file and downloads to your computer
```

---

## 💾 Step 5: Download Trained Model

### What Gets Downloaded

After training completes, you'll download:

```
floormind_trained_model.zip (size: ~3.5GB)
├── final_model/
│   ├── unet/
│   ├── vae/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── scheduler/
│   └── training_config.json
├── floormind_model.pkl
├── training_stats.csv
└── test_generations.png
```

### Download Process

The notebook will automatically:
1. Create a zip file of all outputs
2. Trigger browser download
3. File will be in your Downloads folder

**If download fails:**
```python
# Run this cell to manually download
from google.colab import files
files.download('/content/floormind_trained_model.zip')
```

---

## 📦 Step 6: Integrate Model with Frontend

### Extract Model Files

```bash
# On your local machine
cd /path/to/FloorMind
unzip ~/Downloads/floormind_trained_model.zip -d models/trained_model/
```

### Verify File Structure

Your project should now have:

```
FloorMind/
├── models/
│   └── trained_model/
│       ├── unet/
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── vae/
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── text_encoder/
│       │   ├── config.json
│       │   └── model.safetensors
│       ├── tokenizer/
│       │   ├── tokenizer_config.json
│       │   ├── vocab.json
│       │   └── merges.txt
│       ├── scheduler/
│       │   └── scheduler_config.json
│       ├── model_index.json
│       ├── training_config.json
│       └── training_stats.csv
```

### Update Backend Configuration

The backend is already configured to look for the model at `models/trained_model/`.

Verify in `backend/app.py`:
```python
model_path = "../models/trained_model"  # ✅ Correct path
```

---

## 🚀 Step 7: Test the Integration

### Start Backend

```bash
cd backend
python app.py
```

**Expected output:**
```
🏠 Starting FloorMind Backend Server...
==================================================
⚠️  Model will be loaded on demand via /model/load endpoint
💡 This prevents startup crashes and allows safer model loading

📡 Available endpoints:
   GET  /health - Health check
   GET  /model/info - Model information
   POST /model/load - Load model on demand
   POST /generate - Generate floor plan
   POST /generate/variations - Generate variations
   GET  /presets - Get predefined presets

🚀 Server starting on http://localhost:5001
==================================================
```

### Load Model

```bash
# In a new terminal or use curl
curl -X POST http://localhost:5001/model/load
```

**Expected response:**
```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "model_info": {
    "is_loaded": true,
    "device": "cpu",
    "model_path": "../models/trained_model",
    "resolution": 512,
    "loaded_at": "2024-01-01T12:00:00"
  }
}
```

### Start Frontend

```bash
# In a new terminal
cd frontend
npm start
```

Browser will open at `http://localhost:3000`

### Test Generation

1. Navigate to `http://localhost:3000/generate`
2. Enter: "Modern 3-bedroom apartment with open kitchen"
3. Click "Generate Floor Plan"
4. Wait 2-3 seconds
5. Your AI-generated floor plan appears! 🎉

---

## 📊 Understanding Training Metrics

### Training Stats CSV

Open `models/trained_model/training_stats.csv` to see:

| Column | Description |
|--------|-------------|
| epoch | Training epoch (0-14) |
| step | Global training step |
| train_loss | Training loss (lower is better) |
| val_loss | Validation loss |
| learning_rate | Current learning rate |
| gpu_memory_gb | GPU memory usage |
| timestamp | When metric was recorded |

### Good Training Indicators

✅ **Training is going well if:**
- Loss decreases steadily
- Validation loss follows training loss
- GPU memory stable (~35-38GB)
- No NaN or Inf values

⚠️ **Warning signs:**
- Loss increases or plateaus early
- Validation loss much higher than training loss
- GPU memory keeps increasing
- Frequent OOM errors

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# In the configuration cell, change:
config['train_batch_size'] = 1  # Reduce from 2
config['gradient_accumulation_steps'] = 8  # Increase from 4
```

### Issue 2: Model Not Loading in Backend

**Error:**
```
❌ Model path not found: ../models/trained_model
```

**Solution:**
```bash
# Verify files exist
ls -la models/trained_model/

# Check for required files
ls models/trained_model/unet/
ls models/trained_model/vae/
ls models/trained_model/text_encoder/
```

### Issue 3: Slow Training

**Symptoms:** < 1 step per second

**Solutions:**
1. Verify A100 GPU is selected (not T4)
2. Check GPU utilization: `!nvidia-smi`
3. Reduce `dataloader_num_workers` to 0
4. Enable XFormers (should be automatic)

### Issue 4: Download Fails

**Solution:**
```python
# Save to Google Drive instead
import shutil
shutil.make_archive(
    '/content/drive/MyDrive/FloorMind/floormind_model',
    'zip',
    '/content/outputs/models/floormind_baseline'
)
print("✅ Saved to Google Drive: FloorMind/floormind_model.zip")
```

---

## 📝 Quick Reference

### Essential Commands

```bash
# Check GPU
!nvidia-smi

# Monitor training
# (Progress bar shows automatically)

# Save checkpoint manually
accelerator.save_state(f"{config['output_dir']}/manual_checkpoint")

# Resume from checkpoint
# (Add this before training loop)
accelerator.load_state(f"{config['output_dir']}/checkpoint-1000")
```

### File Sizes

| File/Folder | Size | Description |
|-------------|------|-------------|
| train_images.npy | ~2-5GB | Training images |
| test_images.npy | ~200-500MB | Test images |
| unet/ | ~3.4GB | Fine-tuned UNet model |
| vae/ | ~320MB | VAE encoder/decoder |
| text_encoder/ | ~470MB | CLIP text encoder |
| Total model | ~3.5GB | Complete trained model |

### Training Time Estimates

| Dataset Size | A100-40GB | A100-80GB | T4 |
|--------------|-----------|-----------|-----|
| 1,000 images | ~1 hour | ~45 min | ~3 hours |
| 5,000 images | ~4 hours | ~3 hours | ~12 hours |
| 10,000 images | ~8 hours | ~6 hours | ~24 hours |

---

## ✅ Checklist

Before starting training:
- [ ] Dataset uploaded to Google Drive
- [ ] Colab notebook opened
- [ ] A100 GPU selected
- [ ] Google Drive mounted
- [ ] Dataset files verified

After training:
- [ ] Model downloaded successfully
- [ ] Files extracted to `models/trained_model/`
- [ ] Backend starts without errors
- [ ] Model loads successfully
- [ ] Frontend connects to backend
- [ ] Test generation works

---

## 🆘 Need Help?

- 📖 Full Documentation: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
- 🎓 Training Guide: [docs/guides/TRAINING_GUIDE_A100.md](docs/guides/TRAINING_GUIDE_A100.md)
- 💬 GitHub Issues: [Create an issue](https://github.com/yourusername/FloorMind/issues)
- 📧 Email: support@floormind.ai

---

## 🎉 Success!

Once you see your first AI-generated floor plan, you've successfully:
- ✅ Trained a custom diffusion model
- ✅ Integrated it with your application
- ✅ Created a working AI floor plan generator

**Congratulations! 🎊**

Now you can:
- Generate unlimited floor plans
- Experiment with different prompts
- Fine-tune the model further
- Deploy to production

---

**Happy Training! 🚀**