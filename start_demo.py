#!/usr/bin/env python3
"""
FloorMind Demo Startup Script
Launches the demo interface and backend services
"""

import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    try:
        import flask
        import pandas
        import numpy
        print("✅ Python dependencies available")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    # Check if Node.js is available (optional)
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js available: {result.stdout.strip()}")
        else:
            print("⚠️  Node.js not available (React app won't work)")
    except FileNotFoundError:
        print("⚠️  Node.js not found (React app won't work)")
    
    return True

def start_backend():
    """Start the Flask backend server"""
    print("\n🚀 Starting FloorMind backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    try:
        # Start Flask app in background
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        if process.poll() is None:
            print("✅ Backend server started on http://localhost:5000")
            return process
        else:
            print("❌ Failed to start backend server")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the React frontend (if available)"""
    print("\n🎨 Starting FloorMind frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Failed to install frontend dependencies")
            return None
    
    try:
        # Start React app
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ Frontend server starting on http://localhost:3000")
        return process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def open_demo():
    """Open the demo HTML file"""
    print("\n🌐 Opening FloorMind demo...")
    
    demo_file = Path("frontend/demo.html")
    if demo_file.exists():
        try:
            webbrowser.open(f"file://{demo_file.absolute()}")
            print("✅ Demo opened in browser")
            return True
        except Exception as e:
            print(f"❌ Error opening demo: {e}")
            return False
    else:
        print("❌ Demo file not found")
        return False

def main():
    """Main demo startup function"""
    print("🏠 FloorMind Demo Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Start services
    backend_process = start_backend()
    frontend_process = start_frontend()
    
    # If React frontend failed, open HTML demo
    if frontend_process is None:
        print("\n📄 React frontend not available, opening HTML demo...")
        open_demo()
    else:
        print("\n⏳ Waiting for React frontend to start...")
        time.sleep(5)
        try:
            webbrowser.open("http://localhost:3000")
            print("✅ React frontend opened in browser")
        except:
            print("⚠️  Please manually open http://localhost:3000")
    
    print("\n" + "=" * 50)
    print("🎉 FloorMind Demo is running!")
    print("=" * 50)
    print("\n📍 Available URLs:")
    if backend_process:
        print("  🔧 Backend API: http://localhost:5000")
    if frontend_process:
        print("  🎨 React Frontend: http://localhost:3000")
    else:
        print("  📄 HTML Demo: file://frontend/demo.html")
    
    print("\n💡 Usage:")
    print("  • Try the generator interface")
    print("  • Explore model comparisons")
    print("  • View performance metrics")
    print("  • Read about the project")
    
    print("\n⚠️  Press Ctrl+C to stop all services")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping FloorMind demo...")
        
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
        
        print("👋 Thanks for trying FloorMind!")

if __name__ == "__main__":
    main()