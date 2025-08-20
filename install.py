#!/usr/bin/env python3
"""
Installation script for OctaneDB vector database.

This script helps users install the library and its dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core dependencies
    dependencies = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "h5py>=3.1.0",
        "msgpack>=1.0.0",
        "tqdm>=4.62.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            return False
    
    # Install optional FAISS for enhanced performance
    print("🔧 Installing optional FAISS for enhanced performance...")
    try:
        subprocess.run(f"{sys.executable} -m pip install faiss-cpu>=1.7.0", 
                      shell=True, check=True, capture_output=True)
        print("✅ FAISS installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  FAISS installation failed, continuing without it")
        print("   You can install it later with: pip install faiss-cpu")
    
    return True


def install_development_dependencies():
    """Install development dependencies."""
    print("\n🛠️  Installing development dependencies...")
    
    dev_dependencies = [
        "pytest>=6.0",
        "pytest-cov>=2.10",
        "black>=21.0",
        "flake8>=3.8",
        "psutil>=5.8.0"
    ]
    
    for dep in dev_dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    return True


def install_octanedb():
    """Install OctaneDB in development mode."""
    print("\n🚀 Installing OctaneDB...")
    
    # Check if we're in the right directory
    if not Path("setup.py").exists():
        print("❌ setup.py not found. Please run this script from the OctaneDB root directory.")
        return False
    
    # Install in development mode
    if not run_command(f"{sys.executable} -m pip install -e .", "Installing OctaneDB"):
        return False
    
    return True


def run_tests():
    """Run the test suite to verify installation."""
    print("\n🧪 Running tests to verify installation...")
    
    if not run_command(f"{sys.executable} -m pytest tests/ -v", "Running test suite"):
        print("⚠️  Tests failed, but installation may still be successful")
        return False
    
    return True


def run_example():
    """Run a simple example to verify functionality."""
    print("\n🎯 Running example to verify functionality...")
    
    try:
        # Simple import test
        import octanedb
        print("✅ OctaneDB imported successfully")
        
        # Create a simple database
        db = octanedb.OctaneDB(dimension=64)
        print("✅ Database created successfully")
        
        # Create collection
        collection = db.create_collection("test")
        print("✅ Collection created successfully")
        
        # Insert a vector
        import numpy as np
        vector = np.random.rand(64).astype(np.float32)
        vector_id = db.insert(vector)
        print(f"✅ Vector inserted with ID: {vector_id}")
        
        # Search
        results = db.search(vector, k=1)
        print(f"✅ Search completed, found {len(results)} results")
        
        print("✅ All functionality verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False


def main():
    """Main installation function."""
    print("🚀 OctaneDB Vector Database Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check the errors above.")
        sys.exit(1)
    
    # Install development dependencies (optional)
    install_development_dependencies()
    
    # Install OctaneDB
    if not install_octanedb():
        print("\n❌ Failed to install OctaneDB. Please check the errors above.")
        sys.exit(1)
    
    # Run tests
    run_tests()
    
    # Run example
    run_example()
    
    print("\n🎉 Installation completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Check out the examples/ directory for usage examples")
    print("   2. Read the README.md for detailed documentation")
    print("   3. Run 'python examples/basic_usage.py' to see it in action")
    print("   4. Run 'python examples/performance_benchmark.py' for performance tests")
    
    print("\n🔧 Development setup:")
    print("   - Run 'python -m pytest tests/' to run tests")
    print("   - Run 'black .' to format code")
    print("   - Run 'flake8 .' to check code quality")
    
    print("\n💡 For help and support:")
    print("   - Check the README.md file")
    print("   - Look at the examples/ directory")
    print("   - Run the test suite for verification")


if __name__ == "__main__":
    main()
