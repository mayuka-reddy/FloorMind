"""
Google Gemini API Service for FloorMind
Handles floor plan generation using Gemini's multimodal capabilities
"""

import os
import json
import base64
import io
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from dataclasses import dataclass

@dataclass
class FloorPlanRequest:
    """Floor plan generation request structure"""
    prompt: str
    style: str = "modern"
    rooms: List[str] = None
    dimensions: Tuple[int, int] = (512, 512)
    include_3d: bool = False
    adjacency_rules: Dict = None

@dataclass
class FloorPlanResponse:
    """Floor plan generation response structure"""
    success: bool
    image_data: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict = None
    generation_time: float = 0.0
    error: Optional[str] = None

class GeminiFloorPlanService:
    """Service for generating floor plans using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini service with API key"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.text_model = genai.GenerativeModel('gemini-pro')
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        
        # Floor plan generation prompts
        self.system_prompts = {
            'floor_plan': """You are an expert architectural AI that generates detailed floor plan descriptions and layouts. 
            Create precise, professional floor plans based on user requirements. Focus on:
            - Accurate room proportions and relationships
            - Proper adjacency and flow between spaces
            - Realistic architectural elements (doors, windows, walls)
            - Functional layout optimization
            - Building code compliance considerations""",
            
            '3d_preparation': """You are preparing floor plan data for 3D visualization. 
            Provide detailed spatial information including:
            - Room heights and ceiling details
            - Wall thickness and materials
            - Door and window specifications
            - Furniture placement suggestions
            - Lighting and electrical considerations"""
        }
    
    def generate_floor_plan(self, request: FloorPlanRequest) -> FloorPlanResponse:
        """Generate a floor plan using Gemini API"""
        start_time = time.time()
        
        try:
            # Step 1: Generate detailed floor plan description
            plan_description = self._generate_plan_description(request)
            
            # Step 2: Create visual floor plan
            image_result = self._create_floor_plan_image(plan_description, request)
            
            # Step 3: Enhance with architectural details
            enhanced_metadata = self._enhance_with_architectural_details(
                plan_description, request
            )
            
            # Step 4: Prepare 3D data if requested
            if request.include_3d:
                enhanced_metadata['3d_data'] = self._prepare_3d_data(
                    plan_description, request
                )
            
            generation_time = time.time() - start_time
            
            return FloorPlanResponse(
                success=True,
                image_data=image_result['image_data'],
                image_path=image_result['image_path'],
                metadata=enhanced_metadata,
                generation_time=generation_time
            )
            
        except Exception as e:
            return FloorPlanResponse(
                success=False,
                error=str(e),
                generation_time=time.time() - start_time
            )
    
    def _generate_plan_description(self, request: FloorPlanRequest) -> str:
        """Generate detailed floor plan description using Gemini"""
        
        # Construct enhanced prompt
        enhanced_prompt = f"""
        {self.system_prompts['floor_plan']}
        
        User Request: {request.prompt}
        Style: {request.style}
        Dimensions: {request.dimensions[0]}x{request.dimensions[1]} pixels
        
        Additional Requirements:
        """
        
        if request.rooms:
            enhanced_prompt += f"- Required rooms: {', '.join(request.rooms)}\n"
        
        if request.adjacency_rules:
            enhanced_prompt += f"- Adjacency rules: {json.dumps(request.adjacency_rules)}\n"
        
        enhanced_prompt += """
        
        Please provide a detailed architectural floor plan description including:
        1. Overall layout and orientation
        2. Room dimensions and relationships
        3. Door and window placements
        4. Traffic flow patterns
        5. Functional zones and their connections
        6. Specific architectural elements
        
        Format the response as a structured description that can be used for visual generation.
        """
        
        try:
            response = self.text_model.generate_content(enhanced_prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Failed to generate plan description: {str(e)}")
    
    def _create_floor_plan_image(self, description: str, request: FloorPlanRequest) -> Dict:
        """Create visual floor plan image"""
        
        # For now, we'll create a structured floor plan using PIL
        # In the future, this can be enhanced with Gemini's image generation capabilities
        
        width, height = request.dimensions
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Parse description to extract room information
        rooms_info = self._parse_room_information(description)
        
        # Generate layout based on parsed information
        layout = self._generate_layout(rooms_info, width, height)
        
        # Draw the floor plan
        self._draw_floor_plan(draw, layout, width, height)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"floorplan_gemini_{timestamp}.png"
        image_path = f"../outputs/sample_generations/{filename}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)
        
        # Convert to base64 for API response
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'image_data': image_data,
            'image_path': image_path,
            'layout': layout
        }
    
    def _parse_room_information(self, description: str) -> List[Dict]:
        """Parse room information from Gemini's description"""
        
        # Use Gemini to extract structured data from the description
        parsing_prompt = f"""
        Extract room information from this floor plan description and return as JSON:
        
        {description}
        
        Return a JSON array where each room has:
        - name: room name
        - type: room type (bedroom, bathroom, kitchen, etc.)
        - approximate_size: relative size (small, medium, large)
        - connections: list of connected rooms
        - features: list of special features (windows, doors, etc.)
        
        Example format:
        [
            {{
                "name": "Master Bedroom",
                "type": "bedroom",
                "approximate_size": "large",
                "connections": ["Master Bathroom", "Hallway"],
                "features": ["large window", "walk-in closet"]
            }}
        ]
        """
        
        try:
            response = self.text_model.generate_content(parsing_prompt)
            # Extract JSON from response
            json_start = response.text.find('[')
            json_end = response.text.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response.text[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback to default room structure
                return self._create_default_rooms(description)
                
        except Exception as e:
            print(f"Error parsing room information: {e}")
            return self._create_default_rooms(description)
    
    def _create_default_rooms(self, description: str) -> List[Dict]:
        """Create default room structure if parsing fails"""
        
        # Simple keyword-based room detection
        room_keywords = {
            'bedroom': ['bedroom', 'bed room', 'master', 'guest room'],
            'bathroom': ['bathroom', 'bath', 'toilet', 'restroom'],
            'kitchen': ['kitchen', 'cook', 'culinary'],
            'living_room': ['living', 'lounge', 'family room', 'great room'],
            'dining_room': ['dining', 'eat', 'breakfast'],
            'office': ['office', 'study', 'work', 'den'],
            'hallway': ['hall', 'corridor', 'entry'],
            'closet': ['closet', 'storage', 'pantry']
        }
        
        detected_rooms = []
        description_lower = description.lower()
        
        for room_type, keywords in room_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    detected_rooms.append({
                        'name': room_type.replace('_', ' ').title(),
                        'type': room_type,
                        'approximate_size': 'medium',
                        'connections': [],
                        'features': []
                    })
                    break
        
        return detected_rooms if detected_rooms else [
            {'name': 'Living Room', 'type': 'living_room', 'approximate_size': 'large', 'connections': [], 'features': []},
            {'name': 'Bedroom', 'type': 'bedroom', 'approximate_size': 'medium', 'connections': [], 'features': []},
            {'name': 'Kitchen', 'type': 'kitchen', 'approximate_size': 'medium', 'connections': [], 'features': []},
            {'name': 'Bathroom', 'type': 'bathroom', 'approximate_size': 'small', 'connections': [], 'features': []}
        ]
    
    def _generate_layout(self, rooms_info: List[Dict], width: int, height: int) -> Dict:
        """Generate room layout coordinates"""
        
        layout = {
            'rooms': [],
            'connections': [],
            'dimensions': (width, height)
        }
        
        # Simple grid-based layout algorithm
        num_rooms = len(rooms_info)
        if num_rooms == 0:
            return layout
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_rooms)))
        rows = int(np.ceil(num_rooms / cols))
        
        room_width = width // cols
        room_height = height // rows
        
        for i, room in enumerate(rooms_info):
            row = i // cols
            col = i % cols
            
            x = col * room_width
            y = row * room_height
            
            # Adjust size based on room type and size
            size_multiplier = {
                'small': 0.8,
                'medium': 1.0,
                'large': 1.2
            }.get(room.get('approximate_size', 'medium'), 1.0)
            
            actual_width = int(room_width * size_multiplier)
            actual_height = int(room_height * size_multiplier)
            
            layout['rooms'].append({
                'name': room['name'],
                'type': room['type'],
                'x': x,
                'y': y,
                'width': actual_width,
                'height': actual_height,
                'features': room.get('features', [])
            })
        
        return layout
    
    def _draw_floor_plan(self, draw: ImageDraw.Draw, layout: Dict, width: int, height: int):
        """Draw the floor plan on the image"""
        
        # Color scheme for different room types
        room_colors = {
            'bedroom': '#E3F2FD',
            'bathroom': '#F3E5F5',
            'kitchen': '#E8F5E8',
            'living_room': '#FFF3E0',
            'dining_room': '#FCE4EC',
            'office': '#F1F8E9',
            'hallway': '#FAFAFA',
            'closet': '#EFEBE9'
        }
        
        # Draw rooms
        for room in layout['rooms']:
            x, y = room['x'], room['y']
            w, h = room['width'], room['height']
            
            # Get room color
            color = room_colors.get(room['type'], '#F5F5F5')
            
            # Draw room rectangle
            draw.rectangle([x, y, x + w, y + h], fill=color, outline='#333333', width=2)
            
            # Draw room label
            try:
                # Try to load a font, fallback to default if not available
                font = ImageFont.load_default()
            except:
                font = None
            
            text = room['name']
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(text) * 6
                text_height = 12
            
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2
            
            draw.text((text_x, text_y), text, fill='#333333', font=font)
            
            # Draw doors (simple lines)
            if 'door' in room.get('features', []):
                door_x = x + w // 2
                draw.line([door_x - 10, y, door_x + 10, y], fill='#8B4513', width=3)
            
            # Draw windows (simple rectangles)
            if 'window' in room.get('features', []):
                window_y = y + h // 2
                draw.rectangle([x - 2, window_y - 15, x + 2, window_y + 15], fill='#87CEEB')
    
    def _enhance_with_architectural_details(self, description: str, request: FloorPlanRequest) -> Dict:
        """Enhance floor plan with architectural details using Gemini"""
        
        enhancement_prompt = f"""
        Based on this floor plan description, provide detailed architectural specifications:
        
        {description}
        
        Return a JSON object with:
        1. room_specifications: detailed specs for each room
        2. architectural_elements: doors, windows, walls
        3. building_systems: electrical, plumbing, HVAC considerations
        4. accessibility_features: ADA compliance elements
        5. estimated_metrics: square footage, efficiency ratings
        6. construction_notes: materials and methods
        
        Focus on practical, buildable specifications.
        """
        
        try:
            response = self.text_model.generate_content(enhancement_prompt)
            
            # Try to extract JSON from response
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response.text[json_start:json_end]
                enhanced_data = json.loads(json_str)
            else:
                enhanced_data = {'raw_response': response.text}
            
            # Add generation metadata
            enhanced_data.update({
                'generation_method': 'gemini_enhanced',
                'model_version': 'gemini-pro',
                'timestamp': datetime.now().isoformat(),
                'request_params': {
                    'style': request.style,
                    'dimensions': request.dimensions,
                    'include_3d': request.include_3d
                }
            })
            
            return enhanced_data
            
        except Exception as e:
            return {
                'error': f'Enhancement failed: {str(e)}',
                'generation_method': 'gemini_enhanced',
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_3d_data(self, description: str, request: FloorPlanRequest) -> Dict:
        """Prepare 3D visualization data using Gemini"""
        
        threed_prompt = f"""
        {self.system_prompts['3d_preparation']}
        
        Floor Plan Description: {description}
        
        Generate 3D visualization data as JSON with:
        1. room_heights: ceiling heights for each room
        2. wall_specifications: thickness, materials, structural elements
        3. door_details: types, sizes, swing directions, hardware
        4. window_details: sizes, types, placement heights
        5. furniture_layout: suggested furniture with 3D coordinates
        6. lighting_plan: fixture types and positions
        7. material_palette: colors, textures, finishes
        8. camera_angles: suggested viewpoints for 3D rendering
        
        Provide realistic, architecturally sound specifications.
        """
        
        try:
            response = self.text_model.generate_content(threed_prompt)
            
            # Try to extract JSON from response
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response.text[json_start:json_end]
                threed_data = json.loads(json_str)
            else:
                threed_data = {'raw_response': response.text}
            
            # Add 3D-specific metadata
            threed_data.update({
                'format_version': '1.0',
                'coordinate_system': 'right_handed',
                'units': 'feet',
                'ready_for_3d': True,
                'supported_formats': ['obj', 'fbx', 'gltf', 'usd']
            })
            
            return threed_data
            
        except Exception as e:
            return {
                'error': f'3D preparation failed: {str(e)}',
                'ready_for_3d': False
            }
    
    def analyze_existing_floorplan(self, image_path: str) -> Dict:
        """Analyze an existing floor plan image using Gemini Vision"""
        
        try:
            # Load and prepare image
            image = Image.open(image_path)
            
            # Convert to base64 for Gemini
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            analysis_prompt = """
            Analyze this floor plan image and provide detailed information:
            
            1. Room identification and types
            2. Approximate dimensions and proportions
            3. Door and window locations
            4. Traffic flow patterns
            5. Architectural style and features
            6. Potential improvements or issues
            7. Compliance with building standards
            
            Provide a comprehensive architectural analysis.
            """
            
            # Note: This would use Gemini Vision API when available
            # For now, return a placeholder structure
            
            return {
                'analysis_method': 'gemini_vision',
                'timestamp': datetime.now().isoformat(),
                'image_analyzed': True,
                'placeholder': 'Vision analysis will be implemented with Gemini Vision API'
            }
            
        except Exception as e:
            return {
                'error': f'Image analysis failed: {str(e)}',
                'analysis_method': 'gemini_vision',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_design_suggestions(self, prompt: str, style_preferences: List[str] = None) -> Dict:
        """Get design suggestions and alternatives using Gemini"""
        
        suggestion_prompt = f"""
        Provide creative floor plan design suggestions for: {prompt}
        
        Style preferences: {style_preferences or ['modern', 'functional']}
        
        Generate 3-5 alternative design concepts with:
        1. Layout variations
        2. Room arrangement alternatives
        3. Space optimization ideas
        4. Style-specific features
        5. Pros and cons of each approach
        6. Estimated costs and complexity
        
        Focus on practical, innovative solutions.
        """
        
        try:
            response = self.text_model.generate_content(suggestion_prompt)
            
            return {
                'suggestions': response.text,
                'method': 'gemini_creative',
                'timestamp': datetime.now().isoformat(),
                'style_preferences': style_preferences
            }
            
        except Exception as e:
            return {
                'error': f'Suggestion generation failed: {str(e)}',
                'method': 'gemini_creative',
                'timestamp': datetime.now().isoformat()
            }