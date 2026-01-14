#!/bin/bash
# Setup script for Meta-Cognitive Reasoning project

set -e  # Exit on error

echo "🚀 Setting up Meta-Cognitive Reasoning environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.10+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "📦 Installing package in development mode..."
pip install -e .

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed data/generated
mkdir -p checkpoints logs results

# Create .gitkeep files
touch data/raw/.gitkeep data/processed/.gitkeep data/generated/.gitkeep

# Setup pre-commit hooks (if available)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download datasets: bash scripts/download_data.sh"
echo "  2. Configure wandb: wandb login"
echo "  3. Start training: python scripts/train_teacher.py"
