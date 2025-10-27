#!/usr/bin/env python3
"""
FloorMind Backend Server - Mock Version
Flask server that simulates model responses for testing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
import json
import logging
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_loaded = False
model_info = {
    "is_loaded": False,
    "device": "none",
    "model_path": "./google",
    "resolution": 512
}

def create_mock_floor_plan(description, width=512, height=512):
    """Create a mock floor plan image"""
    
    # Create a white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Define colors
    wall_color = '#2C3E50'
    room_colors = ['#E8F4FD', '#FFF2E8', '#F0F8E8', '#FDF2F8', '#F8F0FF']
    
    # Draw outer walls
    wall_thickness = 8
    draw.rectangle([0, 0, width, wall_thickness], fill=wall_color)  # Top
    draw.rectangle([0, 0, wall_thickness, height], fill=wall_color)  # Left
    draw.rectangle([width-wall_thickness, 0, width, height], fill=wall_color)  # Right
    draw.rectangle([0, height-wall_thickness, width, height], fill=wall_color)  # Bottom
    
    # Parse description to determine layout
    description_lower = description.lower()
    
    # Determine number of rooms based on description
    if 'studio' in description_lower or '1-bedroom' in description_lower:
        rooms = ['Living/Bedroom', 'Kitchen', 'Bathroom']
    elif '2-bedroom' in description_lower or '2 bedroom' in description_lower:
        rooms = ['Living Room', 'Kitchen', 'Bedroom 1', 'Bedroom 2', 'Bathroom']
    elif '3-bedroom' in description_lower or '3 bedroom' in description_lower:
        rooms = ['Living Room', 'Kitchen', 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bathroom']
    elif '4-bedroom' in description_lower or '4 bedroom' in description_lower:
        rooms = ['Living Room', 'Kitchen', 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bedroom 4', 'Bathroom 1', 'Bathroom 2']
    else:
        rooms = ['Living Room', 'Kitchen', 'Bedroom', 'Bathroom']
    
    # Create room layouts
    margin = wall_thickness + 10
    available_width = width - 2 * margin
    available_height = height - 2 * margin
    
    if len(rooms) <= 3:
        # Simple layout for small apartments
        room_width = available_width // 2
        room_height = available_height // 2
        
        positions = [
            (margin, margin, room_width, room_height),
            (margin + room_width, margin, room_width, room_height),
            (margin, margin + room_height, available_width, room_height)
        ]
    elif len(rooms) <= 5:
        # Medium layout
        room_width = available_width // 3
        room_height = available_height // 2
        
        positions = [
            (margin, margin, room_width * 2, room_height),
            (margin + room_width * 2, margin, room_width, room_height),
            (margin, margin + room_height, room_width, room_height),
            (margin + room_width, margin + room_height, room_width, room_height),
            (margin + room_width * 2, margin + room_height, room_width, room_height)
        ]
    else:
        # Large layout
        room_width = available_width // 3
        room_height = available_height // 3
        
        positions = []
        for i in range(min(len(rooms), 9)):
            row = i // 3
            col = i % 3
            x = margin + col * room_width
            y = margin + row * room_height
            positions.append((x, y, room_width, room_height))
    
    # Draw rooms
    for i, room in enumerate(rooms[:len(positions)]):
        if i < len(positions):
            x, y, w, h = positions[i]
            
            # Fill room with color
            color = room_colors[i % len(room_colors)]
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=wall_color, width=2)
            
            # Add room label
            try:
                # Try to use a font, fall back to default if not available
                font_size = max(12, min(w, h) // 8)
                font = ImageFont.load_default()
            except:
                font = None
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), room, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2
            
            draw.text((text_x, text_y), room, fill='#2C3E50', font=font)
    
    # Add some architectural details
    if 'modern' in description_lower:
        # Add some modern elements
        draw.rectangle([margin + 20, margin + 20, margin + 60, margin + 40], fill='#34495E')  # Kitchen island
    
    if 'open' in description_lower:
        # Remove some walls for open concept
        pass  # Already handled by room layout
    
    return img

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "FloorMind AI Backend (Mock)",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify({
        "status": "success",
        "model_info": model_info
    })

@app.route('/model/load', methods=['POST'])
def load_model_endpoint():
    """Simulate loading the model"""
    global model_loaded, model_info
    
    logger.info("🔄 Simulating FloorMind model loading...")
    
    # Simulate loading time
    time.sleep(2)
    
    model_loaded = True
    model_info = {
        "is_loaded": True,
        "device": "cpu",
        "model_path": "./google",
        "resolution": 512,
        "loaded_at": datetime.now().isoformat(),
        "mode": "mock"
    }
    
    logger.info("✅ FloorMind model simulation loaded successfully")
    
    return jsonify({
        "status": "success",
        "message": "Model loaded successfully (Mock Mode)",
        "model_info": model_info
    })

@app.route('/generate', methods=['POST'])
def generate_floor_plan():
    """Generate a mock floor plan"""
    
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded. Please load the model first using /model/load endpoint.",
            "status": "model_not_loaded"
        }), 400
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "Description is required"}), 400
        
        description = data['description']
        
        # Generation parameters
        params = {
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'num_inference_steps': data.get('steps', 20),
            'guidance_scale': data.get('guidance', 7.5),
            'seed': data.get('seed', None)
        }
        
        logger.info(f"🎨 Generating mock floor plan: {description[:50]}...")
        
        # Simulate generation time
        generation_time = random.uniform(2, 5)  # 2-5 seconds
        time.sleep(generation_time)
        
        # Create mock floor plan
        generated_image = create_mock_floor_plan(
            description, 
            params['width'], 
            params['height']
        )
        
        # Convert to base64
        image_data_url = image_to_base64(generated_image)
        
        # Save if requested
        saved_path = None
        if data.get('save', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "./generated_floor_plans"
            os.makedirs(save_dir, exist_ok=True)
            saved_path = os.path.join(save_dir, f"mock_floor_plan_{timestamp}.png")
            generated_image.save(saved_path)
        
        return jsonify({
            "status": "success",
            "description": description,
            "image": image_data_url,
            "parameters": params,
            "saved_path": saved_path,
            "timestamp": datetime.now().isoformat(),
            "generation_time": generation_time,
            "mode": "mock"
        })
        
    except Exception as e:
        logger.error(f"❌ Mock generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/presets', methods=['GET'])
def get_presets():
    """Get predefined floor plan presets"""
    
    presets = {
        "residential": [
            "Modern 3-bedroom apartment with open kitchen and living room",
            "Cozy 2-bedroom house with separate dining area",
            "Spacious studio apartment with efficient layout",
            "Luxury 4-bedroom penthouse with master suite and balcony",
            "Traditional family home with garage and dining room",
            "Contemporary loft with industrial design elements"
        ],
        "commercial": [
            "Small office space with reception area and meeting rooms",
            "Open-plan coworking space with flexible seating",
            "Retail store layout with customer area and storage",
            "Restaurant floor plan with dining area and kitchen"
        ]
    }
    
    return jsonify({
        "status": "success",
        "presets": presets
    })

if __name__ == "__main__":
    print("🏠 Starting FloorMind Backend Server (Mock Mode)...")
    print("=" * 50)
    print("🎭 This is a MOCK server for testing the frontend")
    print("💡 It generates simulated floor plans without loading the actual model")
    print("⚡ Use POST /model/load to 'load' the mock model")
    
    print("\n📡 Available endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /model/info - Model information")
    print("   POST /model/load - Load mock model")
    print("   POST /generate - Generate mock floor plan")
    print("   GET  /presets - Get predefined presets")
    
    print(f"\n🚀 Server starting on http://localhost:5001")
    print("=" * 50)
    
    # Start the server
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 FloorMind mock server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")