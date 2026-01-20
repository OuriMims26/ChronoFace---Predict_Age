#!/bin/bash

################################################################################
# Environment Setup Script for Age Estimation Project
################################################################################
#
# This script automates the complete setup process:
# 1. Installs required Python dependencies
# 2. Downloads UTKFace dataset (if available)
# 3. Extracts and prepares dataset structure
# 4. Verifies GPU availability
# 5. Creates necessary directories
#
# Usage:
#   bash setup_environment.sh
#
# For Google Colab:
#   !bash setup_environment.sh
#
################################################################################

echo "======================================================================"
echo "AGE ESTIMATION PROJECT - ENVIRONMENT SETUP"
echo "======================================================================"
echo ""

# ==============================================================================
# STEP 1: DETECT ENVIRONMENT
# ==============================================================================

echo "======================================================================"
echo "Step 1: Detecting environment..."
echo "======================================================================"

# Check if running in Google Colab
if [ -d "/content" ]; then
    echo "✓ Google Colab environment detected"
    IS_COLAB=true
    PYTHON_CMD="python3"
else
    echo "✓ Local environment detected"
    IS_COLAB=false
    PYTHON_CMD="python"
fi

echo ""

# ==============================================================================
# STEP 2: CHECK GPU AVAILABILITY
# ==============================================================================

echo "======================================================================"
echo "Step 2: Checking GPU availability..."
echo "======================================================================"

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo "⚠️  No GPU detected - training will use CPU (slower)"
    GPU_AVAILABLE=false
fi

echo ""

# ==============================================================================
# STEP 3: INSTALL PYTHON DEPENDENCIES
# ==============================================================================

echo "======================================================================"
echo "Step 3: Installing Python dependencies..."
echo "======================================================================"
echo "This may take a few minutes on first installation"
echo ""

# Upgrade pip to latest version
$PYTHON_CMD -m pip install --upgrade pip -q

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt -q
    echo "✓ Dependencies installed successfully"
else
    echo "⚠️  requirements.txt not found - installing manually..."
    
    # Install core dependencies
    $PYTHON_CMD -m pip install torch torchvision timm -q
    $PYTHON_CMD -m pip install pandas pillow scikit-learn tqdm -q
    
    echo "✓ Core dependencies installed"
fi

echo ""

# ==============================================================================
# STEP 4: CREATE PROJECT DIRECTORIES
# ==============================================================================

echo "======================================================================"
echo "Step 4: Creating project directories..."
echo "======================================================================"

# Create directory structure
mkdir -p data
mkdir -p saved_models
mkdir -p checkpoints
mkdir -p results

echo "✓ Created directories:"
echo "  - data/          (for datasets)"
echo "  - saved_models/  (for trained models)"
echo "  - checkpoints/   (for training checkpoints)"
echo "  - results/       (for predictions and visualizations)"

echo ""

# ==============================================================================
# STEP 5: DATASET PREPARATION (OPTIONAL)
# ==============================================================================

echo "======================================================================"
echo "Step 5: Dataset preparation"
echo "======================================================================"

# Check if dataset already exists
if [ -d "data/utkface_extracted" ]; then
    echo "✓ Dataset already extracted at data/utkface_extracted"
elif [ -f "data/utkface.zip" ] || [ -f "data/dataset.zip" ]; then
    echo "Dataset archive found - extracting..."
    
    # Run Python preparation script
    $PYTHON_CMD prepare_utkface.py
    
    echo "✓ Dataset extraction completed"
else
    echo "⚠️  No dataset found"
    echo ""
    echo "To download UTKFace dataset:"
    echo "1. Visit: https://susanqq.github.io/UTKFace/"
    echo "2. Download the aligned & cropped dataset"
    echo "3. Place the ZIP file in the 'data/' directory"
    echo "4. Run this script again"
    echo ""
    echo "Or for Google Colab, upload to Google Drive and update config.py"
fi

echo ""

# ==============================================================================
# STEP 6: VERIFY INSTALLATION
# ==============================================================================

echo "======================================================================"
echo "Step 6: Verifying installation..."
echo "======================================================================"

# Create verification script
cat > /tmp/verify_install.py << 'EOF'
import sys

try:
    import torch
    import torchvision
    import timm
    import pandas
    import PIL
    import sklearn
    import tqdm
    
    print("✓ All required packages imported successfully")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  TorchVision version: {torchvision.__version__}")
    print(f"  TIMM version: {timm.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    
    sys.exit(0)
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

$PYTHON_CMD /tmp/verify_install.py
VERIFY_STATUS=$?

echo ""

# ==============================================================================
# SETUP COMPLETE
# ==============================================================================

if [ $VERIFY_STATUS -eq 0 ]; then
    echo "======================================================================"
    echo "✅ SETUP COMPLETED SUCCESSFULLY!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "1. Ensure your dataset is in the data/ directory"
    echo "2. Review and adjust config.py if needed"
    echo "3. Start training:"
    echo "   $PYTHON_CMD train_estimator.py"
    echo ""
    echo "For single image prediction:"
    echo "   $PYTHON_CMD inference.py --image path/to/image.jpg --checkpoint saved_models/best_age_estimator.pth"
    echo ""
    echo "======================================================================"
else
    echo "======================================================================"
    echo "⚠️  SETUP COMPLETED WITH WARNINGS"
    echo "======================================================================"
    echo ""
    echo "Some dependencies may not have installed correctly."
    echo "Please check the error messages above."
    echo ""
    echo "======================================================================"
fi