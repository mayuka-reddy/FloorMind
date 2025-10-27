# FloorMind - Complete Project Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation & Setup](#installation--setup)
4. [Training the Model](#training-the-model)
5. [Running the Application](#running-the-application)
6. [API Documentation](#api-documentation)
7. [Frontend Guide](#frontend-guide)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

## Project Overview

**FloorMind** is an AI-powered floor plan generation system that uses fine-tuned Stable Diffusion models to create architectural floor plans from natural language descriptions.

### Key Features
- 🎨 Text-to-floor plan generation using diffusion models
- 🏗️ Constraint-aware architecture for realistic layouts
- 📊 Comprehensive metrics tracking (Accuracy, F1, Loss)
- 🚀 Optimized for Google Colab A100 GPU
- 💻 Modern React frontend with responsive design
- 🔌 RESTful API backend with Flask

### Performance Metrics
- **Accuracy**: 84.5%
- **FID Score**: 57.4
- **CLIP Score**: 0.75
- **Generation Time**: 2.3s average

---

## Project Structure

```
FloorMind/
├── backend/                    # Flask API server
│   ├── app.py                 # Main application
│   ├── routes/                # API endpoints
│   └── services/              # Business logic
│
├── frontend/                   # React application
│   ├── public/                # Static assets
│   ├── src/
│   │   ├── components/        # Reusable components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   └── data/              # JSON data files
│   └── package.json
│
├── notebooks/                  # Jupyter notebooks
│   ├── FloorMind_Colab_Training.ipynb
│   └── CubiCasa5K_Processing_Enhanced.ipynb
│
├── models/                     # Model files
│   ├── trained_model/         # Trained model components
│   ├── checkpoints/           # Training checkpoints
│   └── configs/               # Model configurations
│
├── data/                       # Dataset files
│   ├── processed/             # Processed data
│   └── dataset_manager.py     # Data utilities
│
├── config/                     # Configuration files
│   └── training_config_a100.json
│
├── docs/                       # Documentation
│   └── guides/                # Detailed guides
│       └── TRAINING_GUIDE_A100.md
│
├── scripts/                    # Utility scripts
│   ├── legacy/                # Old scripts
│   ├── testing/               # Test scripts
│   └── utilities/             # Helper scripts
│
├── outputs/                    # Training outputs
│   ├── models/                # Saved models
│   ├── metrics/               # Training metrics
│   └── checkpoints/           # Model checkpoints
│
├── generated_floor_plans/     # Generated images
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Main readme
```

---

## Installation & Setup

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- Node.js 16+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 50GB+ free disk space
```

### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/FloorMind.git
cd FloorMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

---

## Training the Model

### Quick Start (Google Colab)

1. **Open the Training Notebook**
   - Upload `notebooks/FloorMind_Colab_Training.ipynb` to Google Colab
   - Select Runtime > Change runtime type > A100 GPU

2. **Upload Dataset**
   - Prepare your dataset as numpy arrays
   - Upload files when prompted:
     - `train_images.npy`
     - `train_descriptions.npy`
     - `test_images.npy`
     - `test_descriptions.npy`

3. **Run Training**
   - Execute all cells sequentially
   - Monitor training progress
   - Download trained model when complete

### Training Configuration

The optimized configuration for A100 is located at `config/training_config_a100.json`:

```json
{
  "train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "num_epochs": 15,
  "learning_rate": 5e-6,
  "mixed_precision": "fp16",
  "use_8bit_adam": true
}
```

### Expected Training Time

| Dataset Size | Epochs | A100-40GB | A100-80GB |
|--------------|--------|-----------|-----------|
| 1,000 images | 15     | ~1 hour   | ~45 min   |
| 5,000 images | 15     | ~4 hours  | ~3 hours  |
| 10,000 images| 15     | ~8 hours  | ~6 hours  |

### Metrics Tracked

- **Training Loss**: MSE between predicted and actual noise
- **Validation Loss**: Performance on held-out data
- **Learning Rate**: Current learning rate value
- **GPU Memory**: Memory usage in GB
- **Gradient Norm**: Gradient magnitude for stability monitoring

For detailed training instructions, see [`docs/guides/TRAINING_GUIDE_A100.md`](docs/guides/TRAINING_GUIDE_A100.md).

---

## Running the Application

### Start Backend Server

```bash
# From project root
cd backend
python app.py

# Server will start on http://localhost:5001
```

### Start Frontend

```bash
# From project root
cd frontend
npm start

# Application will open at http://localhost:3000
```

### Using Docker (Optional)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access application at http://localhost:3000
```

---

## API Documentation

### Base URL
```
http://localhost:5001
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "FloorMind AI Backend",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": true
}
```

#### 2. Model Information
```http
GET /model/info
```

**Response:**
```json
{
  "status": "success",
  "model_info": {
    "is_loaded": true,
    "device": "cuda",
    "resolution": 512,
    "loaded_at": "2024-01-01T12:00:00"
  }
}
```

#### 3. Load Model
```http
POST /model/load
```

**Response:**
```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "model_info": {...}
}
```

#### 4. Generate Floor Plan
```http
POST /generate
```

**Request Body:**
```json
{
  "description": "Modern 3-bedroom apartment with open kitchen",
  "width": 512,
  "height": 512,
  "steps": 20,
  "guidance": 7.5,
  "seed": 42,
  "save": true
}
```

**Response:**
```json
{
  "status": "success",
  "description": "Modern 3-bedroom apartment with open kitchen",
  "image": "data:image/png;base64,...",
  "parameters": {...},
  "saved_path": "generated_floor_plans/floor_plan_20240101_120000.png",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 5. Generate Variations
```http
POST /generate/variations
```

**Request Body:**
```json
{
  "description": "Cozy 2-bedroom house",
  "variations": 4,
  "width": 512,
  "height": 512,
  "steps": 20,
  "guidance": 7.5,
  "seed": 42
}
```

**Response:**
```json
{
  "status": "success",
  "description": "Cozy 2-bedroom house",
  "variations": [
    {
      "variation": 1,
      "image": "data:image/png;base64,...",
      "seed": 42
    },
    ...
  ],
  "parameters": {...},
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 6. Get Presets
```http
GET /presets
```

**Response:**
```json
{
  "status": "success",
  "presets": {
    "residential": [...],
    "commercial": [...],
    "architectural_styles": [...]
  }
}
```

---

## Frontend Guide

### Pages

1. **Home Page** (`/`)
   - Project overview
   - Key features
   - Performance statistics
   - Call-to-action

2. **Generator Page** (`/generate`)
   - Text input for descriptions
   - Generation parameters
   - Real-time preview
   - Download options

3. **Models Page** (`/models`)
   - Model architecture
   - Training details
   - Performance metrics

4. **Metrics Page** (`/metrics`)
   - Training curves
   - Evaluation metrics
   - Comparison charts

5. **About Page** (`/about`)
   - Project information
   - Technical details
   - Use cases

6. **Developers Page** (`/developers`)
   - Team members
   - Contributions
   - Technology stack
   - Acknowledgments

### Responsive Design

The frontend is fully responsive and optimized for:
- 📱 **Mobile**: 320px - 767px
- 📱 **Tablet**: 768px - 1023px
- 💻 **Desktop**: 1024px+

### Customization

#### Update Developer Information

Edit `frontend/src/data/developers.json`:

```json
{
  "developers": [
    {
      "id": 1,
      "name": "Your Name",
      "role": "Your Role",
      "avatar": "URL to avatar",
      "bio": "Your bio",
      "contributions": ["Contribution 1", "Contribution 2"],
      "links": {
        "github": "https://github.com/username",
        "linkedin": "https://linkedin.com/in/username",
        "email": "your.email@example.com"
      }
    }
  ]
}
```

#### Update Styling

Modify `frontend/tailwind.config.js` for theme customization:

```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {...},
        secondary: {...}
      }
    }
  }
}
```

---

## Development Workflow

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add: your feature description"

# Push to remote
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Code Style

- **Python**: Follow PEP 8
- **JavaScript**: Use ESLint configuration
- **Commits**: Use conventional commits format

### Testing

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
```python
# Reduce batch size in config
config['train_batch_size'] = 1
config['gradient_accumulation_steps'] = 8
```

#### 2. Model Not Loading

**Solution:**
```bash
# Check model path
ls -la models/trained_model/

# Verify model files exist
# Required: model.safetensors, config.json, etc.
```

#### 3. Frontend Not Connecting to Backend

**Solution:**
```javascript
// Check API URL in frontend/src/services/api.js
const API_URL = 'http://localhost:5001';
```

#### 4. Slow Training

**Solution:**
```python
# Enable XFormers
unet.enable_xformers_memory_efficient_attention()

# Reduce num_workers if I/O bound
config['dataloader_num_workers'] = 0
```

### Getting Help

- 📧 Email: support@floormind.ai
- 💬 GitHub Issues: [Create an issue](https://github.com/yourusername/FloorMind/issues)
- 📖 Documentation: Check `docs/guides/`

---

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](docs/guides/CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- 🎨 New constraint types
- 📊 Dataset improvements
- 🔧 Performance optimization
- 🎯 Evaluation metrics
- 📱 Frontend enhancements

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Stability AI** - Stable Diffusion base model
- **Hugging Face** - Diffusers library
- **CubiCasa** - CubiCasa5K dataset
- **PyTorch Team** - Deep learning framework

---

## Citation

If you use FloorMind in your research, please cite:

```bibtex
@software{floormind2024,
  title={FloorMind: AI-Powered Floor Plan Generation},
  author={Your Team},
  year={2024},
  url={https://github.com/yourusername/FloorMind}
}
```

---

**Built with ❤️ by the FloorMind Team**

For more information, visit our [GitHub repository](https://github.com/yourusername/FloorMind).