# Release Checklist

This document provides a step-by-step guide for creating and publishing a new release of the Risk Control project using `uv`.

## 📋 Pre-Release Checklist

Before starting the release process, ensure you have completed the following:

### ✅ Code Quality Checks
```bash
# Run all quality checks
make lint
make type-check
make tests
make report-tests
```

### ✅ Documentation
```bash
# Build and verify documentation
make docs
make serve  # Check locally if needed
```

### ✅ Dependencies
- [ ] All dependencies are up to date
- [ ] Development dependencies are properly configured

### ✅ Git Status
- [ ] All changes are committed
- [ ] Working directory is clean
- [ ] You're on the main branch (or release branch)

## 🚀 Release Process

### Step 1: Update Version

Choose your version update method:

#### Option A: Set Specific Version
```bash
uv version 1.0.0
```

#### Option B: Bump Version Automatically
```bash
# Bump patch version (0.1.0 → 0.1.1)
uv version --bump patch

# Bump minor version (0.1.0 → 0.2.0)
uv version --bump minor

# Bump major version (0.1.0 → 1.0.0)
uv version --bump major
```

### Step 2: Build Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
uv build
```

This creates:
- Source distribution (`.tar.gz`)
- Wheel distribution (`.whl`)
- Both files are placed in the `dist/` directory

### Step 3: Verify Build

```bash
# Check what was built
ls -la dist/

# Verify package contents (optional)
tar -tzf dist/riskcontrol-1.0.0.tar.gz
```

### Step 4: Test Package Locally (Optional)

```bash
# Install the built package in a test environment
pip install dist/riskcontrol-1.0.0.whl

# Test basic functionality
python -c "import risk_control; print('Package works!')"
```

### Step 5: Commit and Tag

```bash
# Add all changes
git add .

# Commit with conventional commit message
git commit -m "release: version 1.0.0"

# Create and push tag
git tag v1.0.0
git push origin main
git push origin v1.0.0
```

### Step 7: Publish to PyPI

⚠️ **WARNING**: This will publish to PyPI and cannot be undone!

```bash
# Publish to PyPI
uv publish
```

**Prerequisites:**
- PyPI account with API token
- Token configured in your environment
- Package name available on PyPI

## 🔧 Configuration

### PyPI Authentication

Set up your PyPI credentials:

```bash
# Using API token (recommended)
export UV_INDEX_URL=https://pypi.org/simple/
export UV_INDEX_USERNAME=__token__
export UV_INDEX_PASSWORD=pypi-your-token-here

# Or configure in ~/.pypirc
```

### Test PyPI (Optional)

For testing releases, you can publish to Test PyPI first:

```bash
# Build for Test PyPI
uv build

# Publish to Test PyPI
uv publish --index-url https://test.pypi.org/simple/
```

## 📦 Package Information

### Current Package Details
- **Name**: `riskcontrol`
- **Description**: Calibrate predictive algorithms to achieve risk control
- **Python Version**: >=3.13
- **License**: BSD 3-Clause

### Distribution Files
- **Source Distribution**: `riskcontrol-{version}.tar.gz`
- **Wheel Distribution**: `riskcontrol-{version}-py3-none-any.whl`

## 🧪 Post-Release Verification

After publishing, verify the release:

1. **Check PyPI**: Visit https://pypi.org/project/riskcontrol/
2. **Test Installation**: `pip install riskcontrol`
3. **Verify Documentation**: Check if docs are updated
4. **Test Examples**: Run example scripts with new version

## 🔄 Rollback (If Needed)

If you need to rollback a release:

1. **Delete the tag**:
   ```bash
   git tag -d v1.0.0
   git push origin :refs/tags/v1.0.0
   ```

2. **Revert version in pyproject.toml**:
   ```bash
   uv version 0.1.0  # or previous version
   ```

3. **Note**: PyPI releases cannot be deleted, only deprecated

## 📚 Additional Resources

- [uv Package Guide](https://docs.astral.sh/uv/guides/package/)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PyPI Help](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)

## 🎯 Quick Release Commands

For experienced users, here's the minimal workflow:

```bash
# 1. Quality checks
make lint && make type-check && make tests

# 2. Update version
uv version --bump minor

# 3. Build and publish
uv build
git add . && git commit -m "release: version $(uv version)"
git tag v$(uv version)
git push origin main --tags
uv publish
```

---

**Remember**: Always test thoroughly before releasing, and consider using Test PyPI for first-time releases!
