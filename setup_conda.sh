#!/bin/bash
# Conda-based setup script for H100 server

set -e

echo "=========================================="
echo "Setting up metacog-reasoning with Conda"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the metacog-reasoning directory"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo ""
    echo "To install Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    exit 1
fi

# Environment name
ENV_NAME="metacog"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Conda environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment..."
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Activate environment
echo ""
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "✓ Conda environment '${ENV_NAME}' activated"

# Detect CUDA version
echo ""
echo "Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p')
    if [ -z "$CUDA_VERSION" ]; then
        echo "Warning: Could not detect CUDA version. Assuming CUDA 11.8"
        CUDA_VERSION="11.8"
    else
        echo "Found CUDA $CUDA_VERSION"
    fi
else
    echo "Warning: nvidia-smi not found. Assuming CUDA 11.8"
    CUDA_VERSION="11.8"
fi

# Set PyTorch installation command based on CUDA version
if [[ "$CUDA_VERSION" == 12.* ]]; then
    TORCH_INSTALL="pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
    echo "Using PyTorch with CUDA 12.1 support"
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    TORCH_INSTALL="pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo "Using PyTorch with CUDA 11.8 support"
else
    TORCH_INSTALL="pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo "Using PyTorch with CUDA 11.8 support (default)"
fi

# Set environment variables
echo ""
echo "Setting environment variables..."
export HF_HOME=~/hf_cache
export TRANSFORMERS_CACHE=~/hf_cache
export HF_DATASETS_CACHE=~/hf_cache/datasets

# Add to bashrc for persistence
echo ""
echo "Adding environment variables to ~/.bashrc..."
grep -q "HF_HOME" ~/.bashrc || echo 'export HF_HOME=~/hf_cache' >> ~/.bashrc
grep -q "TRANSFORMERS_CACHE" ~/.bashrc || echo 'export TRANSFORMERS_CACHE=~/hf_cache' >> ~/.bashrc
grep -q "HF_DATASETS_CACHE" ~/.bashrc || echo 'export HF_DATASETS_CACHE=~/hf_cache/datasets' >> ~/.bashrc

# Add conda activation to bashrc
if ! grep -q "conda activate ${ENV_NAME}" ~/.bashrc; then
    echo ""
    echo "Adding conda activation to ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# Auto-activate metacog environment" >> ~/.bashrc
    echo "# conda activate ${ENV_NAME}" >> ~/.bashrc
    echo "(Commented out - uncomment to auto-activate)"
fi

# Create cache directories
echo ""
echo "Creating cache directories..."
mkdir -p ~/hf_cache/datasets
mkdir -p ~/hf_cache/hub
mkdir -p logs
mkdir -p data/benchmarks

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support via conda
echo ""
echo "Installing PyTorch with CUDA support..."
conda install ${TORCH_INSTALL} -y

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo ""
echo "Installing package in development mode..."
pip install -e .

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.sh
chmod +x scripts/*.py 2>/dev/null || true

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA info not available"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "GPU count not available"

if command -v nvidia-smi &> /dev/null; then
    python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
fi

python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=========================================="
    echo "GPU Information:"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Conda environment '${ENV_NAME}' is now active."
echo ""
echo "To activate this environment in future sessions:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Download benchmarks:"
echo "     bash scripts/download_benchmarks.sh"
echo ""
echo "  2. Run quick test (10 samples):"
echo "     python scripts/evaluate_baseline.py --max_samples 10"
echo ""
echo "  3. Run full evaluation:"
echo "     python scripts/evaluate_baseline.py"
echo ""
echo "To use specific GPU:"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_baseline.py"
echo "  CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_baseline.py"
echo ""
