#!/usr/bin/env python3
"""
FloorMind Complete Launcher v2.0
Restructured project with proper organization
"""

import os
import sys
import subprocess
import time
import threading
import signal
import requests

class FloorMindLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        self.project_root = self._find_project_root()
    
    def _find_project_root(self):
        """Find the project root directory"""
        
        current = os.path.abspath(".")
        
        # Look for key project files
        key_files = ["package.json", "requirements.txt", "README.md"]
        
        while current != "/":
            if any(os.path.exists(os.path.join(current, f)) for f in key_files):
                return current
            current = os.path.dirname(current)
        
        return os.path.abspath(".")
    
    def check_requirements(self):
        """Check all requirements"""
        
        print("🔍 Checking system requirements...")
        
        # Check Python packages
        python_packages = ['flask', 'flask-cors', 'torch', 'diffusers', 'transformers', 'safetensors', 'pillow']
        missing_python = []
        
        for package in python_packages:
            try:
                # Special handling for pillow
                if package == 'pillow':
                    import PIL
                else:
                    __import__(package.replace('-', '_'))
                print(f"✅ Python: {package}")
            except ImportError:
                print(f"❌ Python: {package}")
                missing_python.append(package)
        
        # Check Node.js
        node_ok = False
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Node.js: {result.stdout.strip()}")
                node_ok = True
            else:
                print("❌ Node.js: Not found")
        except FileNotFoundError:
            print("❌ Node.js: Not found")
        
        # Check npm
        npm_ok = False
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ npm: {result.stdout.strip()}")
                npm_ok = True
            else:
                print("❌ npm: Not found")
        except FileNotFoundError:
            print("❌ npm: Not found")
        
        # Report results
        if missing_python:
            print(f"\n❌ Missing Python packages: {missing_python}")
            print("💡 Install with: pip install " + " ".join(missing_python))
            return False
        
        if not node_ok:
            print("\n❌ Node.js is required for the frontend")
            print("💡 Install from https://nodejs.org/")
            return False
        
        if not npm_ok:
            print("\n❌ npm is required for the frontend")
            return False
        
        print("✅ All requirements satisfied!")
        return True
    
    def check_project_structure(self):
        """Check if project structure is correct"""
        
        print("\n🔍 Checking project structure...")
        
        required_paths = [
            "src/api/app.py",
            "src/core/model_manager.py",
            "frontend/package.json",
            "frontend/src"
        ]
        
        missing_paths = []
        for path in required_paths:
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                print(f"✅ {path}")
            else:
                print(f"❌ {path}")
                missing_paths.append(path)
        
        if missing_paths:
            print(f"\n⚠️  Missing paths: {missing_paths}")
            print("💡 Project structure may need to be updated")
            return False
        
        print("✅ Project structure looks good!")
        return True
    
    def install_frontend_deps(self):
        """Install frontend dependencies"""
        
        print("\n📦 Checking frontend dependencies...")
        
        frontend_path = os.path.join(self.project_root, "frontend")
        if not os.path.exists(frontend_path):
            print(f"❌ Frontend directory not found: {frontend_path}")
            return False
        
        node_modules_path = os.path.join(frontend_path, "node_modules")
        if os.path.exists(node_modules_path):
            print("✅ Frontend dependencies already installed")
            return True
        
        print("📦 Installing frontend dependencies...")
        try:
            result = subprocess.run(
                ['npm', 'install'], 
                cwd=frontend_path,
                check=True, 
                capture_output=True, 
                text=True
            )
            print("✅ Frontend dependencies installed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install frontend dependencies: {e}")
            return False
    
    def start_backend(self):
        """Start the backend server"""
        
        print("\n🚀 Starting backend server...")
        
        api_app_path = os.path.join(self.project_root, "src", "api", "app.py")
        
        if not os.path.exists(api_app_path):
            print(f"❌ Backend app not found: {api_app_path}")
            return False
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, api_app_path],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if backend is running
            if self.backend_process.poll() is None:
                print("✅ Backend server started successfully!")
                return True
            else:
                print("❌ Backend server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready"""
        
        print("⏳ Waiting for backend to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:5001/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend is ready!")
                    return True
            except:
                pass
            
            time.sleep(1)
        
        print("❌ Backend failed to become ready")
        return False
    
    def start_frontend(self):
        """Start the frontend server"""
        
        print("\n🎨 Starting frontend server...")
        
        frontend_path = os.path.join(self.project_root, "frontend")
        
        try:
            # Set environment variable to avoid browser auto-opening
            env = os.environ.copy()
            env['BROWSER'] = 'none'
            
            self.frontend_process = subprocess.Popen(
                ['npm', 'start'],
                cwd=frontend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Wait a moment for startup
            time.sleep(5)
            
            # Check if frontend is running
            if self.frontend_process.poll() is None:
                print("✅ Frontend server started successfully!")
                return True
            else:
                print("❌ Frontend server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor both processes"""
        
        print("\n👀 Monitoring processes...")
        print("💡 Press Ctrl+C to stop both servers")
        print("=" * 50)
        print("🌐 Frontend: http://localhost:3000")
        print("🔧 Backend:  http://localhost:5001")
        print("📚 API Docs: http://localhost:5001/")
        print("=" * 50)
        
        try:
            while self.running:
                # Check backend
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend process died!")
                    break
                
                # Check frontend
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ Frontend process died!")
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 Shutting down FloorMind...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown both processes"""
        
        self.running = False
        
        if self.backend_process:
            print("🔄 Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing backend...")
                self.backend_process.kill()
        
        if self.frontend_process:
            print("🔄 Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing frontend...")
                self.frontend_process.kill()
        
        print("👋 FloorMind shutdown complete!")
    
    def run(self):
        """Main run method"""
        
        print("🏠 FloorMind Complete Launcher v2.0")
        print("=" * 50)
        print(f"📍 Project root: {self.project_root}")
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Requirements check failed!")
            return
        
        # Check project structure
        if not self.check_project_structure():
            print("\n⚠️  Project structure issues detected")
            print("💡 Some features may not work correctly")
        
        # Install frontend dependencies
        if not self.install_frontend_deps():
            print("\n❌ Frontend setup failed!")
            return
        
        # Start backend
        if not self.start_backend():
            print("\n❌ Backend startup failed!")
            return
        
        # Wait for backend to be ready
        if not self.wait_for_backend():
            print("\n❌ Backend not ready!")
            self.shutdown()
            return
        
        # Start frontend
        if not self.start_frontend():
            print("\n❌ Frontend startup failed!")
            self.shutdown()
            return
        
        # Monitor processes
        self.monitor_processes()

def main():
    launcher = FloorMindLauncher()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        launcher.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        launcher.run()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        launcher.shutdown()

if __name__ == "__main__":
    main()