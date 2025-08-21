# 🚀 OctaneDB PyPI Publishing Guide

This guide will walk you through publishing OctaneDB to the Python Package Index (PyPI) so users can install it with `pip install octanedb`.

## 📋 Prerequisites

### **1. PyPI Account Setup**

#### **Test PyPI (Recommended for first time)**
1. Go to [Test PyPI](https://test.pypi.org/account/register/)
2. Create an account with username and email
3. Verify your email address
4. Note your username and password

#### **Production PyPI**
1. Go to [PyPI](https://pypi.org/account/register/)
2. Create an account with username and email
3. Verify your email address
4. **Enable 2FA** (recommended for security)
5. Note your username and password

### **2. Install Required Tools**

```bash
pip install twine build
```

### **3. Configure PyPI Credentials**

Create a `.pypirc` file in your home directory:

**Windows:**
```bash
# Create file: C:\Users\YourUsername\.pypirc
```

**macOS/Linux:**
```bash
# Create file: ~/.pypirc
```

**File content:**
```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = your_test_username
password = your_test_password

[pypi]
repository = https://upload.pypi.org/legacy/
username = your_production_username
password = your_production_password
```

**⚠️ Security Note:** For production, consider using API tokens instead of passwords:
```ini
[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-your_api_token_here
```

## 🚀 Publishing Workflow

### **Option 1: Automated Publishing (Recommended)**

Use the provided publishing script:

```bash
python publish_to_pypi.py
```

The script will guide you through:
1. ✅ Prerequisites check
2. 🧹 Clean build files
3. 🔨 Build package
4. 🔍 Check package
5. 🚀 Upload to PyPI
6. ✅ Verify installation

### **Option 2: Manual Publishing**

#### **Step 1: Update Package Information**

Before publishing, update these files:

1. **`setup.py`** - Update author email and GitHub URL
2. **`CHANGELOG.md`** - Ensure version matches
3. **`README.md`** - Verify all links work

#### **Step 2: Clean Previous Builds**

```bash
# Remove previous build artifacts
rm -rf build/ dist/ *.egg-info/
```

#### **Step 3: Build Package**

```bash
# Build source distribution
python -m build --sdist

# Build wheel distribution
python -m build --wheel
```

#### **Step 4: Check Package**

```bash
# Verify package integrity
twine check dist/*
```

#### **Step 5: Upload to Test PyPI (Recommended First)**

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ octanedb
```

#### **Step 6: Upload to Production PyPI**

```bash
# Upload to Production PyPI
twine upload dist/*

# Verify installation
pip install octanedb
```

## 📦 Package Structure

Your package structure should look like this:

```
OctaneDB/
├── octanedb/
│   ├── __init__.py
│   ├── core.py
│   ├── collection.py
│   ├── index.py
│   ├── storage.py
│   ├── query.py
│   ├── utils.py
│   └── cli.py
├── tests/
├── examples/
├── setup.py
├── MANIFEST.in
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .pypirc
```

## 🔧 Configuration Files

### **setup.py**
- ✅ Package metadata
- ✅ Dependencies
- ✅ Classifiers
- ✅ Entry points (CLI)

### **MANIFEST.in**
- ✅ Include source files
- ✅ Include documentation
- ✅ Exclude build artifacts

### **requirements.txt**
- ✅ Core dependencies
- ✅ Version constraints
- ✅ No development dependencies

## 🧪 Testing Before Publishing

### **1. Local Installation Test**

```bash
# Install in development mode
pip install -e .

# Test import
python -c "import octanedb; print('✅ Import successful')"

# Test CLI
octanedb info
```

### **2. Package Build Test**

```bash
# Build package
python -m build

# Check package
twine check dist/*

# List package contents
tar -tzf dist/*.tar.gz
```

### **3. Test PyPI Upload**

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ octanedb

# Test functionality
python -c "import octanedb; db = octanedb.OctaneDB(384); print('✅ Test successful')"
```

## 🚀 Publishing Checklist

Before publishing, ensure:

- [ ] **Version updated** in `setup.py` and `CHANGELOG.md`
- [ ] **Author information** updated in `setup.py`
- [ ] **GitHub URLs** updated in `setup.py`
- [ ] **Dependencies** correctly specified in `requirements.txt`
- [ ] **README.md** is comprehensive and accurate
- [ ] **Tests pass** locally
- [ ] **Package builds** without errors
- [ ] **Package checks** pass with `twine check`
- [ ] **Test PyPI upload** successful
- [ ] **Test installation** works from Test PyPI

## 📊 After Publishing

### **1. Monitor PyPI**
- Check [PyPI project page](https://pypi.org/project/octanedb/)
- Monitor download statistics
- Check for any reported issues

### **2. Update Documentation**
- Add PyPI badge to README
- Update installation instructions
- Share on social media/GitHub

### **3. Version Management**
- Tag release in Git: `git tag v1.0.0`
- Push tags: `git push --tags`
- Plan next release

## 🔄 Updating Package

For future updates:

1. **Update version** in `setup.py` and `CHANGELOG.md`
2. **Update CHANGELOG.md** with new features/fixes
3. **Test locally** with new version
4. **Build and upload** new version
5. **Tag new release** in Git

## 🆘 Troubleshooting

### **Common Issues**

#### **Authentication Errors**
```bash
# Check .pypirc file
cat ~/.pypirc

# Test credentials
twine check --repository testpypi dist/*
```

#### **Package Build Errors**
```bash
# Clean build files
rm -rf build/ dist/ *.egg-info/

# Rebuild
python -m build
```

#### **Import Errors After Installation**
```bash
# Check package structure
pip show octanedb

# Verify files
python -c "import octanedb; print(octanedb.__file__)"
```

#### **Version Conflicts**
```bash
# Uninstall previous version
pip uninstall octanedb

# Install new version
pip install octanedb
```

## 📚 Additional Resources

- [PyPI Help](https://pypi.org/help/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Test PyPI](https://test.pypi.org/)

## 🎉 Success!

Once published, users can install OctaneDB with:

```bash
pip install octanedb
```

And use it in their code:

```python
from octanedb import OctaneDB

db = OctaneDB(dimension=384)
collection = db.create_collection("my_collection")
```

---

**Need Help?** Open an issue on GitHub or check the troubleshooting section above.
