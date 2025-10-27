# FloorMind Project Structure v2.0

This document describes the new, organized project structure for FloorMind.

## Overview

The project has been restructured from a flat folder layout to a proper hierarchical organization with clear separation of concerns.

## New Directory Structure

```
floormind/
├── src/                          # Source code (new organized structure)
│   ├── core/                     # Core functionality
│   │   ├── model_manager.py      # Centralized model management
│   │   └── __init__.py
│   ├── api/                      # Backend API
│   │   ├── app.py               # Flask application
│   │   ├── routes.py            # API routes
│   │   └── __init__.py
│   ├── frontend/                 # Frontend utilities (if needed)
│   │   ├── services/
│   │   │   └── api.js           # Enhanced API service
│   │   └── __init__.py
│   └── scripts/                  # Utility scripts
│       ├── start_backend.py      # Backend startup
│       ├── start_complete.py     # Complete launcher
│       └── __init__.py
├── frontend/                     # React frontend (existing)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   │   └── api.js           # Updated API service
│   │   └── ...
│   ├── public/
│   └── package.json
├── models/                       # Model storage (organized)
│   ├── trained/                  # Trained models
│   └── checkpoints/              # Training checkpoints
├── data/                         # Dataset and processing (existing)
├── training/                     # Training scripts (existing)
├── outputs/                      # Generated outputs (organized)
│   ├── generated/                # Generated floor plans
│   ├── logs/                     # Application logs
│   └── exports/                  # Exported data
├── docs/                         # Documentation
├── tests/                        # Test files
└── legacy/                       # Old flat structure files (to be moved)
```

## Key Improvements

### 1. Centralized Model Management (`src/core/model_manager.py`)

**Features:**
- Automatic model path detection
- Multiple loading strategies with fallbacks
- Memory optimization
- Comprehensive error handling
- Global model manager instance

**Usage:**
```python
from src.core.model_manager import get_model_manager

manager = get_model_manager()
success = manager.load_model()
image = manager.generate_floor_plan("Modern apartment")
```

### 2. Enhanced API Structure (`src/api/`)

**Features:**
- Clean separation of routes and app logic
- Blueprint-based organization
- Enhanced error handling
- New endpoints (batch generation, model unloading)
- Better logging and monitoring

**New Endpoints:**
- `POST /model/unload` - Free memory by unloading model
- `POST /generate/batch` - Generate multiple floor plans
- Enhanced error responses and validation

### 3. Improved Frontend Integration (`frontend/src/services/api.js`)

**Features:**
- Connection status tracking
- Progress callbacks for long operations
- Enhanced error handling
- Request/response timing
- Automatic retry logic
- Better caching

**Usage:**
```javascript
import floorMindAPI from './services/api';

// Load model with progress
await floorMindAPI.loadModel((progress) => {
  console.log(progress.message);
});

// Generate with enhanced parameters
const result = await floorMindAPI.generateFloorPlan({
  description: "Modern apartment",
  style: "contemporary",
  width: 512,
  height: 512
});
```

### 4. Smart Startup Scripts (`src/scripts/`)

**Features:**
- Automatic dependency checking
- Project structure validation
- Process monitoring
- Graceful shutdown
- Enhanced error reporting

## Migration from Flat Structure

### Files to Move/Update

1. **Model Files:**
   - Move `google/` → `models/trained/`
   - Update all references to new path

2. **Backend Files:**
   - `backend/app.py` → Use `src/api/app.py`
   - `backend/floormind_generator.py` → Integrated into `src/core/model_manager.py`

3. **Scripts:**
   - `start_*.py` → Use `src/scripts/start_complete.py`
   - `test_*.py` → Move to `tests/`

4. **Outputs:**
   - `generated_floor_plans/` → `outputs/generated/`

### Backward Compatibility

The new structure maintains backward compatibility:
- Old API endpoints still work
- Frontend can use either old or new API service
- Model paths are auto-detected

## Usage Instructions

### Quick Start (New Structure)

1. **Start Everything:**
   ```bash
   python src/scripts/start_complete.py
   ```

2. **Backend Only:**
   ```bash
   python src/scripts/start_backend.py
   ```

3. **Frontend Only:**
   ```bash
   cd frontend && npm start
   ```

### Development Workflow

1. **Model Development:**
   - Place trained models in `models/trained/`
   - Use `src/core/model_manager.py` for loading
   - Test with enhanced API endpoints

2. **Frontend Development:**
   - Use updated `frontend/src/services/api.js`
   - Leverage new progress callbacks and error handling
   - Test connection status features

3. **API Development:**
   - Add routes to `src/api/routes.py`
   - Use model manager for model operations
   - Follow blueprint pattern

## Benefits of New Structure

### 1. **Better Organization**
- Clear separation of concerns
- Easier to navigate and maintain
- Follows Python/React best practices

### 2. **Enhanced Reliability**
- Centralized model management
- Better error handling
- Automatic fallbacks and recovery

### 3. **Improved Developer Experience**
- Smart startup scripts
- Better logging and monitoring
- Enhanced debugging capabilities

### 4. **Scalability**
- Modular architecture
- Easy to add new features
- Clean API design

### 5. **Production Ready**
- Proper error handling
- Memory management
- Process monitoring
- Graceful shutdown

## Configuration

### Environment Variables

```bash
# API Configuration
REACT_APP_API_URL=http://localhost:5001

# Model Configuration
FLOORMIND_MODEL_PATH=models/trained

# Logging
LOG_LEVEL=INFO
```

### Model Path Priority

1. Environment variable `FLOORMIND_MODEL_PATH`
2. `models/trained/`
3. `google/` (legacy)
4. Auto-detection in parent directories

## Testing

### Backend Tests
```bash
python -m pytest tests/api/
python tests/test_model_manager.py
```

### Frontend Tests
```bash
cd frontend && npm test
```

### Integration Tests
```bash
python tests/test_integration_v2.py
```

## Troubleshooting

### Common Issues

1. **Model Not Found:**
   - Check `models/trained/` directory
   - Verify model files exist
   - Check console logs for path detection

2. **API Connection Issues:**
   - Ensure backend is running on port 5001
   - Check CORS configuration
   - Verify frontend API URL

3. **Memory Issues:**
   - Use model unloading endpoint
   - Check system resources
   - Enable memory optimizations

### Debug Mode

```bash
# Backend with debug logging
LOG_LEVEL=DEBUG python src/api/app.py

# Frontend with verbose logging
REACT_APP_DEBUG=true npm start
```

## Next Steps

1. **Complete Migration:**
   - Move remaining files to new structure
   - Update all path references
   - Test thoroughly

2. **Enhanced Features:**
   - Add model versioning
   - Implement caching layer
   - Add metrics and monitoring

3. **Production Deployment:**
   - Docker containers
   - Environment-specific configs
   - CI/CD pipeline

This new structure provides a solid foundation for continued development and production deployment of FloorMind.