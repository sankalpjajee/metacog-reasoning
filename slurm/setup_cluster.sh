#!/bin/bash
# Setup script for cluster environment
# Run this once after cloning the repository

set -e

echo "=========================================="
echo "Setting up metacog-reasoning on cluster"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the metacog-reasoning directory"
    exit 1
fi

# Load modules
echo "Loading modules..."
module load cuda/11.8
module load python/3.10
module load gcc/9.3.0

# Set environment variables
echo "Setting environment variables..."
export HF_HOME=/work/sjajee/hf_cache
export TRANSFORMERS_CACHE=/work/sjajee/hf_cache
export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets

# Add to bashrc for persistence
echo ""
echo "Adding environment variables to ~/.bashrc..."
grep -q "HF_HOME" ~/.bashrc || echo 'export HF_HOME=/work/sjajee/hf_cache' >> ~/.bashrc
grep -q "TRANSFORMERS_CACHE" ~/.bashrc || echo 'export TRANSFORMERS_CACHE=/work/sjajee/hf_cache' >> ~/.bashrc
grep -q "HF_DATASETS_CACHE" ~/.bashrc || echo 'export HF_DATASETS_CACHE=/work/sjajee/hf_cache/datasets' >> ~/.bashrc

# Create cache directories
echo "Creating cache directories..."
mkdir -p /work/sjajee/hf_cache/datasets
mkdir -p /work/sjajee/hf_cache/hub
mkdir -p logs

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
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
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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
chmod +x scripts/*.py
chmod +x slurm/*.sh

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA not available (this is OK if not on GPU node)"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Request GPU allocation:"
echo "     salloc --partition=v100-32gb-hiprio --time=08:00:00 --gres=gpu:1"
echo ""
echo "  2. Download benchmarks:"
echo "     sbatch slurm/download_benchmarks.sh"
echo "     OR (if in interactive session):"
echo "     bash scripts/download_benchmarks.sh"
echo ""
echo "  3. Run quick test:"
echo "     sbatch slurm/test_evaluation.sh"
echo ""
echo "  4. Run full evaluation:"
echo "     sbatch slurm/evaluate_baseline.sh"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
