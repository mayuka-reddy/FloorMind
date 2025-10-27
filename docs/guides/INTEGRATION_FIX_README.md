# FloorMind Integration Fix

This document explains the fixes applied to integrate the FloorMind model with the frontend.

## Issues Fixed

### 1. Backend Model Loading
- **Problem**: Model loading was causing segmentation faults during startup
- **Fix**: 
  - Changed to on-demand model loading via `/model/load` endpoint
  - Added safer model loading with fallback mechanisms
  - Improved error handling and memory management
  - Added attention slicing for memory efficiency

### 2. API Integration
- **Problem**: Frontend couldn't properly communicate with backend
- **Fix**:
  - Added proper CORS handling
  - Improved error messages and connection handling
  - Added model loading endpoint for frontend to trigger
  - Better timeout handling for long operations

### 3. Startup Process
- **Problem**: Complex startup process with potential failures
- **Fix**:
  - Created separate startup scripts for backend and frontend
  - Added comprehensive launcher script
  - Proper dependency checking
  - Graceful error handling and shutdown

## New Files Created

### Startup Scripts
- `start_backend.py` - Safe backend startup with dependency checks
- `start_frontend.py` - Frontend startup with npm dependency management
- `start_floormind_fixed.py` - Complete launcher for both services

### Testing Scripts
- `test_backend_simple.py` - Basic backend endpoint testing
- `test_integration_fixed.py` - Complete integration testing

## How to Use

### Option 1: Start Everything Together
```bash
python start_floormind_fixed.py
```

### Option 2: Start Separately
```bash
# Terminal 1 - Backend
python start_backend.py

# Terminal 2 - Frontend  
python start_frontend.py
```

### Option 3: Manual Start
```bash
# Backend
cd backend
python app.py

# Frontend (in another terminal)
cd frontend
npm install
npm start
```

## Usage Flow

1. **Start the servers** using one of the methods above
2. **Open browser** to http://localhost:3000
3. **Load the model** by clicking "Load AI Model" button in the UI
4. **Generate floor plans** by entering descriptions and clicking generate

## Key Improvements

### Backend (`backend/app.py`)
- On-demand model loading prevents startup crashes
- Better error handling and logging
- Memory-efficient model loading
- Proper CORS configuration
- Model loading endpoint for frontend control

### Frontend (`frontend/src/services/api.js`)
- Better error handling for connection issues
- Improved timeout management
- Clear error messages for users
- Proper model loading integration

### Startup Process
- Comprehensive dependency checking
- Automatic npm dependency installation
- Process monitoring and graceful shutdown
- Clear status reporting

## Troubleshooting

### Backend Issues
- **Model loading fails**: Check that model files exist in `google/` directory
- **Segmentation fault**: Use the on-demand loading instead of startup loading
- **Port conflicts**: Backend uses port 5001 to avoid macOS AirPlay conflicts

### Frontend Issues
- **Cannot connect**: Ensure backend is running on port 5001
- **Dependencies missing**: Run `npm install` in frontend directory
- **Port conflicts**: Frontend uses port 3000 by default

### Integration Issues
- **CORS errors**: Backend includes proper CORS headers
- **Timeout errors**: Model loading and generation can take time
- **Memory issues**: Model uses attention slicing for efficiency

## Testing

Run the integration test to verify everything works:
```bash
python test_integration_fixed.py
```

This will test:
- Backend health and endpoints
- Model loading capability
- Frontend accessibility
- Optional generation testing

## Architecture

```
Frontend (React)     Backend (Flask)      Model (Diffusers)
     |                      |                     |
     |-- API calls -------->|                     |
     |                      |-- Load model ------>|
     |                      |                     |
     |<-- JSON response ----|<-- Generated image--|
```

The integration now properly handles:
- Asynchronous model loading
- Error propagation and user feedback
- Memory management
- Process lifecycle management

## Next Steps

1. **Performance optimization**: Add model caching and faster inference
2. **Advanced features**: Batch generation, style transfer, etc.
3. **Production deployment**: Docker containers, environment configs
4. **Monitoring**: Add metrics and logging for production use