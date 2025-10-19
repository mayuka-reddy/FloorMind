# Google Gemini API Integration Setup

This guide explains how to set up Google Gemini API for FloorMind's advanced floor plan generation and 3D preparation features.

## 🔑 Getting Your Gemini API Key

### Step 1: Access Google AI Studio
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Accept the terms of service if prompted

### Step 2: Create API Key
1. Click "Create API Key"
2. Select your Google Cloud project (or create a new one)
3. Copy the generated API key
4. **Important**: Keep this key secure and never commit it to version control

### Step 3: Configure FloorMind
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file and add your API key:
   ```bash
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 🚀 Features Enabled by Gemini

### 1. Advanced Floor Plan Generation
- **Natural Language Understanding**: Better interpretation of complex architectural descriptions
- **Style Awareness**: Generate plans in specific architectural styles
- **Contextual Intelligence**: Understanding of spatial relationships and building codes

### 2. 3D Preparation Data
- **Room Heights**: Ceiling specifications for each room
- **Wall Details**: Thickness, materials, structural elements
- **Furniture Layout**: Suggested 3D furniture placement
- **Lighting Plan**: Fixture types and positions
- **Material Palette**: Colors, textures, finishes for 3D rendering

### 3. Architectural Analysis
- **Image Analysis**: Analyze existing floor plans using Gemini Vision
- **Code Compliance**: Check against building standards
- **Improvement Suggestions**: AI-powered design recommendations

### 4. Design Suggestions
- **Alternative Layouts**: Multiple design variations
- **Style Recommendations**: Architecture style suggestions
- **Optimization Ideas**: Space efficiency improvements

## 📡 API Endpoints

### Generate Floor Plan with Gemini
```bash
POST /api/generate
{
  "prompt": "3-bedroom modern apartment with open kitchen",
  "model_type": "gemini",
  "style": "modern",
  "include_3d": true,
  "seed": 42
}
```

### Generate 3D-Ready Floor Plan
```bash
POST /api/3d-ready
{
  "prompt": "luxury penthouse with panoramic views",
  "style": "contemporary",
  "seed": 123
}
```

### Analyze Existing Floor Plan
```bash
POST /api/analyze
Content-Type: multipart/form-data
image: [floor_plan_image.png]
```

### Get Design Suggestions
```bash
POST /api/suggestions
{
  "prompt": "small studio apartment for young professional",
  "style_preferences": ["minimalist", "scandinavian"]
}
```

### Get Available Models
```bash
GET /api/models
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_api_key

# Optional
DEFAULT_MODEL=gemini
ENABLE_3D_FEATURES=true
GENERATION_TIMEOUT=30
MAX_BATCH_SIZE=10
```

### Model Selection
- `gemini`: Google Gemini API (recommended)
- `constraint_aware`: Enhanced diffusion model
- `baseline`: Standard diffusion model

### Architectural Styles
- `modern`: Clean lines, open spaces
- `contemporary`: Current design trends
- `traditional`: Classic architectural elements
- `minimalist`: Simple, uncluttered design
- `industrial`: Raw materials, exposed elements
- `scandinavian`: Light, functional, cozy

## 🎯 Usage Examples

### Basic Generation
```python
import requests

response = requests.post('http://localhost:5000/api/generate', json={
    'prompt': '2-bedroom apartment with balcony',
    'model_type': 'gemini',
    'style': 'modern'
})

result = response.json()
print(f"Generated: {result['image_path']}")
```

### 3D-Ready Generation
```python
response = requests.post('http://localhost:5000/api/3d-ready', json={
    'prompt': 'family home with garage and garden',
    'style': 'traditional'
})

result = response.json()
if result['success']:
    print("3D data available:", result['metadata']['3d_data'])
```

### Design Analysis
```python
with open('existing_floorplan.png', 'rb') as f:
    response = requests.post('http://localhost:5000/api/analyze', 
                           files={'image': f})

analysis = response.json()
print("Analysis:", analysis)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. API Key Not Working
```bash
# Check if key is set
echo $GEMINI_API_KEY

# Verify key format (should start with 'AI...')
# Regenerate key if needed
```

#### 2. Import Errors
```bash
# Install required packages
pip install google-generativeai>=0.3.0

# Verify installation
python -c "import google.generativeai as genai; print('✅ Gemini installed')"
```

#### 3. Generation Failures
- Check internet connection
- Verify API key permissions
- Check rate limits (Gemini has usage quotas)
- Review prompt complexity

#### 4. 3D Data Issues
- Ensure `include_3d=true` in request
- Use `gemini` model type
- Check response metadata for 3D data

### Error Messages

#### "Gemini API key not found"
```bash
# Set environment variable
export GEMINI_API_KEY=your_key_here

# Or add to .env file
echo "GEMINI_API_KEY=your_key_here" >> .env
```

#### "Gemini API not available"
- Check internet connection
- Verify API key validity
- Check Google AI Studio for service status

## 📊 Performance & Limits

### Generation Times
- **Simple floor plans**: 2-5 seconds
- **Complex layouts**: 5-10 seconds
- **3D-ready plans**: 8-15 seconds

### API Limits
- **Rate limit**: 60 requests per minute
- **Daily quota**: Check Google AI Studio
- **Image size**: Max 4MB for analysis
- **Prompt length**: Max 8,000 characters

### Optimization Tips
1. **Batch requests** when possible
2. **Cache results** for repeated prompts
3. **Use specific prompts** for better results
4. **Monitor usage** in Google AI Studio

## 🔮 Future 3D Integration

The 3D data generated by Gemini is designed for future integration with:

### 3D Rendering Engines
- **Three.js**: Web-based 3D visualization
- **Blender**: Professional 3D modeling
- **Unity**: Interactive 3D experiences
- **Unreal Engine**: Photorealistic rendering

### File Formats
- **OBJ**: Basic 3D geometry
- **FBX**: Advanced 3D with materials
- **GLTF**: Web-optimized 3D
- **USD**: Universal scene description

### Planned Features
- **Real-time 3D preview**: Interactive floor plan exploration
- **VR walkthrough**: Virtual reality experiences
- **AR visualization**: Augmented reality overlay
- **Photorealistic rendering**: High-quality 3D images

## 🆘 Support

### Documentation
- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Docs](https://ai.google.dev/)
- [FloorMind GitHub](https://github.com/your-repo/floormind)

### Community
- GitHub Issues for bug reports
- Discussions for feature requests
- Stack Overflow for technical questions

---

**Ready to generate amazing floor plans with AI! 🏗️✨**