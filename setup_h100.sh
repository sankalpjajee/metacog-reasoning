#!/bin/bash
# Quick setup script for H100 server (non-SLURM environment)

set -e

echo "=========================================="
echo "Setting up metacog-reasoning on H100"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the metacog-reasoning directory"
    exit 1
fi

# Detect CUDA version
echo ""
echo "Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Found CUDA $CUDA_VERSION"
else
    echo "Warning: nvcc not found. Assuming CUDA 11.8"
    CUDA_VERSION="11.8"
fi

# Set PyTorch index URL based on CUDA version
if [[ "$CUDA_VERSION" == 12.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    echo "Using PyTorch with CUDA 12.1 support"
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "Using PyTorch with CUDA 11.8 support"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
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

# Create cache directories
echo "Creating cache directories..."
mkdir -p ~/hf_cache/datasets
mkdir -p ~/hf_cache/hub
mkdir -p logs
mkdir -p data/benchmarks

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "Virtual environment already exists, skipping..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url $TORCH_INDEX

# Install dependencies (without problematic packages)
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
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Show GPU info
echo ""
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
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
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To use specific GPU:"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_baseline.py"
echo "  CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_baseline.py"
echo ""
