# FloorMind - Quick Start Guide

Get FloorMind up and running in 5 minutes! 🚀

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Git

## 🚀 Quick Setup

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/yourusername/FloorMind.git
cd FloorMind

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Start Backend

```bash
cd backend
python app.py
```

Backend will start at `http://localhost:5001`

### 3. Start Frontend

```bash
# In a new terminal
cd frontend
npm start
```

Frontend will open at `http://localhost:3000`

## 🎨 Generate Your First Floor Plan

1. Navigate to `http://localhost:3000/generate`
2. Enter a description: "Modern 3-bedroom apartment with open kitchen"
3. Click "Generate Floor Plan"
4. Wait 2-3 seconds for your AI-generated floor plan!

## 📊 Training Your Own Model (Google Colab)

### Option 1: Use Pre-trained Model
Download from [releases](https://github.com/yourusername/FloorMind/releases) and place in `models/trained_model/`

### Option 2: Train from Scratch

1. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `notebooks/FloorMind_Colab_Training.ipynb`

2. **Select A100 GPU**
   - Runtime > Change runtime type > A100 GPU

3. **Upload Dataset**
   - Prepare CubiCasa5K as numpy arrays
   - Upload when prompted:
     - `train_images.npy`
     - `train_descriptions.npy`
     - `test_images.npy`
     - `test_descriptions.npy`

4. **Run Training**
   - Execute all cells
   - Training takes ~4 hours for 5K images
   - Download model when complete

5. **Deploy Model**
   - Extract downloaded model to `models/trained_model/`
   - Restart backend server

## 🎯 Key Features to Try

### 1. Text-to-Floor Plan Generation
```
Input: "Luxury penthouse with panoramic views"
Output: Detailed architectural floor plan
```

### 2. Multiple Variations
Generate 4 different layouts from the same description

### 3. Custom Parameters
- Adjust inference steps (10-50)
- Modify guidance scale (5-15)
- Set custom seed for reproducibility

### 4. Preset Prompts
Try pre-configured prompts for:
- Residential layouts
- Commercial spaces
- Architectural styles

## 📱 Mobile Access

FloorMind is fully responsive! Access from:
- 📱 Phone (320px+)
- 📱 Tablet (768px+)
- 💻 Desktop (1024px+)

## 🔧 Configuration

### Backend Port
Edit `backend/app.py`:
```python
app.run(host='0.0.0.0', port=5001)
```

### Frontend API URL
Edit `frontend/src/services/api.js`:
```javascript
const API_URL = 'http://localhost:5001';
```

### Model Path
Edit `backend/app.py`:
```python
model_path = "../models/trained_model"
```

## 📚 Next Steps

- 📖 Read [Full Documentation](PROJECT_DOCUMENTATION.md)
- 🎓 Follow [Training Guide](docs/guides/TRAINING_GUIDE_A100.md)
- 👥 Check [Developers Page](http://localhost:3000/developers)
- 📊 View [Metrics Dashboard](http://localhost:3000/metrics)

## ❓ Troubleshooting

### Backend won't start
```bash
# Check if port 5001 is available
lsof -i :5001

# Try different port
python app.py --port 5002
```

### Frontend can't connect
```bash
# Verify backend is running
curl http://localhost:5001/health

# Check CORS settings in backend/app.py
```

### Model not loading
```bash
# Verify model files exist
ls -la models/trained_model/

# Check backend logs for errors
```

## 🆘 Get Help

- 💬 [GitHub Issues](https://github.com/yourusername/FloorMind/issues)
- 📧 Email: support@floormind.ai
- 📖 [Full Documentation](PROJECT_DOCUMENTATION.md)

## 🎉 Success!

You're now ready to generate AI-powered floor plans! 

Visit `http://localhost:3000` to start creating.

---

**Happy Generating! 🏠✨**