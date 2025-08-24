#!/usr/bin/env python3
"""
Modern Python package build and publish script for OctaneDB.
Uses modern tools like build, twine, and setuptools_scm.
"""
"""
Hello
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    required_tools = [
        ("python", "Python interpreter"),
        ("pip", "Python package installer"),
        ("twine", "Python package uploader"),
    ]
    
    missing_tools = []
    
    for tool, description in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(f"{tool} ({description})")
    
    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("💡 Install missing tools with:")
        print("   pip install build twine")
        sys.exit(1)
    
    print("✅ All prerequisites are available")

def clean_build_files():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build files...")
    
    build_dirs = ["build", "dist", "*.egg-info"]
    
    for pattern in build_dirs:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                print(f"   Removing directory: {path}")
                shutil.rmtree(path, ignore_errors=True)
            else:
                print(f"   Removing file: {path}")
                path.unlink(missing_ok=True)
    
    print("✅ Build files cleaned")

def install_dependencies():
    """Install build dependencies."""
    print("📦 Installing build dependencies...")
    
    run_command(
        "pip install --upgrade build twine setuptools_scm[toml]",
        "Installing build tools"
    )
    
    print("✅ Build dependencies installed")

def build_package():
    """Build the package using modern tools."""
    print("🔨 Building package...")
    
    # Build source distribution and wheel
    result = run_command(
        "python -m build",
        "Building source distribution and wheel"
    )
    
    if result.returncode == 0:
        print("✅ Package built successfully")
        
        # List built files
        dist_dir = Path("dist")
        if dist_dir.exists():
            print("📁 Built files:")
            for file in dist_dir.iterdir():
                print(f"   - {file.name}")
    else:
        print("❌ Package build failed")
        sys.exit(1)

def check_package():
    """Check the built package for common issues."""
    print("🔍 Checking package...")
    
    # Check source distribution
    sdist_files = list(Path("dist").glob("*.tar.gz"))
    if sdist_files:
        run_command(
            f"twine check {sdist_files[0]}",
            "Checking source distribution"
        )
    
    # Check wheel
    wheel_files = list(Path("dist").glob("*.whl"))
    if wheel_files:
        run_command(
            f"twine check {wheel_files[0]}",
            "Checking wheel"
        )
    
    print("✅ Package check completed")

def test_install():
    """Test installing the package locally."""
    print("🧪 Testing package installation...")
    
    # Find the wheel file
    wheel_files = list(Path("dist").glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel file found for testing")
        return
    
    wheel_file = wheel_files[0]
    
    # Test install in a temporary environment
    run_command(
        f"pip install {wheel_file} --force-reinstall",
        "Testing package installation"
    )
    
    # Test import
    try:
        import octanedb
        print(f"✅ Package imported successfully: {octanedb.__version__}")
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        sys.exit(1)

def upload_to_pypi(test=False):
    """Upload the package to PyPI."""
    if test:
        print("🚀 Uploading to Test PyPI...")
        repo = "--repository testpypi"
        url = "https://test.pypi.org/legacy/"
    else:
        print("🚀 Uploading to Production PyPI...")
        repo = ""
        url = "https://upload.pypi.org/legacy/"
    
    # Upload using twine
    result = run_command(
        f"twine upload {repo} dist/*",
        f"Uploading to {'Test PyPI' if test else 'Production PyPI'}"
    )
    
    if result.returncode == 0:
        print(f"✅ Package uploaded successfully to {'Test PyPI' if test else 'Production PyPI'}")
        if test:
            print(f"🔗 Test PyPI URL: https://test.pypi.org/project/octanedb/")
        else:
            print(f"🔗 PyPI URL: https://pypi.org/project/octanedb/")
    else:
        print(f"❌ Package upload failed")
        sys.exit(1)

def main():
    """Main build and publish workflow."""
    print("🚀 OctaneDB Package Builder")
    print("=" * 50)
    
    # Check prerequisites
    check_prerequisites()
    
    # Clean previous builds
    clean_build_files()
    
    # Install dependencies
    install_dependencies()
    
    # Build package
    build_package()
    
    # Check package
    check_package()
    
    # Test installation
    test_install()
    
    print("\n🎉 Package build completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review the built files in the 'dist' directory")
    print("2. Test the package locally")
    print("3. Upload to Test PyPI: python build_package.py --test")
    print("4. Upload to Production PyPI: python build_package.py --publish")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            upload_to_pypi(test=True)
        elif sys.argv[1] == "--publish":
            upload_to_pypi(test=False)
        elif sys.argv[1] == "--help":
            print("\n📖 Usage:")
            print("   python build_package.py          # Build package only")
            print("   python build_package.py --test   # Build and upload to Test PyPI")
            print("   python build_package.py --publish # Build and upload to Production PyPI")
            print("   python build_package.py --help   # Show this help message")

if __name__ == "__main__":
    main()
