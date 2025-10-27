#!/usr/bin/env python3
"""
FloorMind Backend Startup Script v2.0
Enhanced startup with proper structure
"""

import os
import sys
import subprocess
import time

def check_requirements():
    """Check if required packages are installed"""
    
    print("🔍 Checking Python requirements...")
    
    required_packages = [
        'flask',
        'flask-cors', 
        'torch',
        'diffusers',
        'transformers',
        'safetensors',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Special handling for pillow
            if package == 'pillow':
                import PIL
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("💡 Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All Python requirements satisfied!")
    return True

def check_model_files():
    """Check if model files exist"""
    
    print("\n🔍 Checking model files...")
    
    # Check multiple possible model locations
    possible_paths = [
        "models/trained",
        "google",
        "../google",
        "../../google"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            required_files = [
                "model.safetensors",
                "tokenizer_config.json", 
                "scheduler_config.json"
            ]
            
            all_files_exist = True
            for file in required_files:
                if not os.path.exists(os.path.join(path, file)):
                    all_files_exist = False
                    break
            
            if all_files_exist:
                model_path = path
                break
    
    if model_path:
        print(f"✅ Model files found in: {model_path}")
        return True
    else:
        print("⚠️  Model files not found in standard locations")
        print("💡 Model will be loaded on-demand if available")
        return False

def start_backend():
    """Start the backend server"""
    
    print("\n🚀 Starting FloorMind Backend v2.0...")
    
    # Find the API app
    api_app_path = None
    possible_paths = [
        "src/api/app.py",
        "../src/api/app.py",
        "../../src/api/app.py"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            api_app_path = path
            break
    
    if not api_app_path:
        print("❌ Could not find src/api/app.py")
        print("💡 Make sure you're running from the project root")
        return False
    
    try:
        # Start the Flask server
        print(f"📍 Using API app: {api_app_path}")
        subprocess.run([sys.executable, api_app_path], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Backend server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Backend server failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Main startup function"""
    
    print("🏠 FloorMind Backend Startup v2.0")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("💡 Please install missing packages and try again")
        return
    
    # Check model files
    check_model_files()
    
    # Start backend
    print("\n" + "=" * 50)
    start_backend()

if __name__ == "__main__":
    main()