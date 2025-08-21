#!/usr/bin/env python3
"""
OctaneDB PyPI Publishing Script
Automates the process of publishing OctaneDB to PyPI
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            print(f"   Stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"   Stderr: {e.stderr.strip()}")
        return False

def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    required_tools = [
        ("python", "Python interpreter"),
        ("pip", "Python package installer"),
        ("twine", "PyPI upload tool"),
        ("build", "Python build tool"),
    ]
    
    missing_tools = []
    
    for tool, description in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(f"{tool} ({description})")
    
    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("\nInstall missing tools with:")
        print("  pip install twine build")
        return False
    
    print("✅ All prerequisites are available")
    return True

def clean_build_files():
    """Clean up previous build files."""
    print("🧹 Cleaning up previous build files...")
    
    build_dirs = ["build", "dist", "*.egg-info"]
    
    for pattern in build_dirs:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   Removed: {path}")
            elif path.is_file():
                path.unlink()
                print(f"   Removed: {path}")
    
    print("✅ Build files cleaned up")

def build_package():
    """Build the Python package."""
    print("🔨 Building Python package...")
    
    # Build source distribution
    if not run_command("python -m build --sdist", "Building source distribution"):
        return False
    
    # Build wheel distribution
    if not run_command("python -m build --wheel", "Building wheel distribution"):
        return False
    
    print("✅ Package built successfully")
    return True

def check_package():
    """Check the built package for issues."""
    print("🔍 Checking built package...")
    
    # Check source distribution
    if not run_command("twine check dist/*", "Checking source distribution"):
        return False
    
    print("✅ Package check completed successfully")
    return True

def test_upload(test_pypi=True):
    """Upload to PyPI (test or production)."""
    if test_pypi:
        print("🚀 Uploading to Test PyPI...")
        repository = "https://test.pypi.org/legacy/"
        print("   This is a test upload - package will not be available on main PyPI")
    else:
        print("🚀 Uploading to Production PyPI...")
        repository = "https://upload.pypi.org/legacy/"
        print("   ⚠️  This will make the package publicly available!")
    
    # Confirm upload
    if not test_pypi:
        confirm = input("\n🤔 Are you sure you want to upload to production PyPI? (yes/no): ")
        if confirm.lower() != "yes":
            print("❌ Upload cancelled")
            return False
    
    # Upload to PyPI
    if not run_command(f"twine upload --repository {repository} dist/*", "Uploading package"):
        return False
    
    print("✅ Package uploaded successfully!")
    
    if test_pypi:
        print("\n📋 Test PyPI URL: https://test.pypi.org/project/octanedb/")
        print("📋 Test installation: pip install --index-url https://test.pypi.org/simple/ octanedb")
    else:
        print("\n📋 Production PyPI URL: https://pypi.org/project/octanedb/")
        print("📋 Production installation: pip install octanedb")
    
    return True

def verify_installation(test_pypi=True):
    """Verify the package can be installed."""
    print("🔍 Verifying package installation...")
    
    if test_pypi:
        install_cmd = "pip install --index-url https://test.pypi.org/simple/ octanedb"
    else:
        install_cmd = "pip install octanedb"
    
    if not run_command(install_cmd, "Installing package"):
        return False
    
    # Test import
    if not run_command("python -c 'import octanedb; print(f\"✅ OctaneDB {octanedb.__version__} imported successfully\")'", "Testing import"):
        return False
    
    print("✅ Package installation verified successfully")
    return True

def show_next_steps():
    """Show next steps after successful upload."""
    print("\n🎉 Congratulations! OctaneDB has been published to PyPI!")
    print("\n📋 Next Steps:")
    print("1. 🐛 Monitor for any issues or bugs")
    print("2. 📚 Update documentation if needed")
    print("3. 🚀 Announce the release on social media/GitHub")
    print("4. 📊 Monitor download statistics on PyPI")
    print("5. 🔄 Plan next release and version bump")
    
    print("\n🔗 Useful Links:")
    print(f"   - PyPI Project: https://pypi.org/project/octanedb/")
    print(f"   - GitHub Repository: https://github.com/yourusername/OctaneDB")
    print(f"   - Documentation: https://github.com/yourusername/OctaneDB#readme")
    
    print("\n💡 Tips:")
    print("   - Users can now install with: pip install octanedb")
    print("   - Consider adding badges to your README")
    print("   - Monitor PyPI analytics for usage insights")

def main():
    """Main publishing workflow."""
    print("🚀 OctaneDB PyPI Publishing Script")
    print("=" * 60)
    print("This script will publish OctaneDB to PyPI")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please install missing tools.")
        return 1
    
    # Ask user what they want to do
    print("\n🤔 What would you like to do?")
    print("1. Test upload to Test PyPI (recommended for first time)")
    print("2. Upload to Production PyPI")
    print("3. Just build and check package (no upload)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Test PyPI workflow
        print("\n🧪 Starting Test PyPI workflow...")
        
        if not clean_build_files():
            return 1
        
        if not build_package():
            return 1
        
        if not check_package():
            return 1
        
        if not test_upload(test_pypi=True):
            return 1
        
        if not verify_installation(test_pypi=True):
            return 1
        
        print("\n✅ Test PyPI workflow completed successfully!")
        print("📋 You can now test the package installation from Test PyPI")
        
    elif choice == "2":
        # Production PyPI workflow
        print("\n🚀 Starting Production PyPI workflow...")
        
        if not clean_build_files():
            return 1
        
        if not build_package():
            return 1
        
        if not check_package():
            return 1
        
        if not test_upload(test_pypi=False):
            return 1
        
        if not verify_installation(test_pypi=False):
            return 1
        
        show_next_steps()
        
    elif choice == "3":
        # Just build and check
        print("\n🔨 Building and checking package only...")
        
        if not clean_build_files():
            return 1
        
        if not build_package():
            return 1
        
        if not check_package():
            return 1
        
        print("\n✅ Package built and checked successfully!")
        print("📋 Package files are in the 'dist/' directory")
        print("📋 You can manually upload them later with: twine upload dist/*")
        
    else:
        print("❌ Invalid choice. Please run the script again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
