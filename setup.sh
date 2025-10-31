#!/bin/bash

# Setup script for PTQ vs QAT VLM Research Project

echo "======================================"
echo "Setting up Qvlm Research Environment"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p experiments/{results,logs,checkpoints}
mkdir -p data/{raw,processed,calibration}
mkdir -p models/weights
mkdir -p docs/{papers,notes}

# Create __init__.py files for Python packages
echo "Creating package files..."
touch src/__init__.py
touch src/ptq/__init__.py
touch src/qat/__init__.py
touch src/evaluation/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/utils/__init__.py

# Create placeholder files for logs
touch experiments/logs/.gitkeep
touch experiments/results/.gitkeep
touch experiments/checkpoints/.gitkeep

# Create research log
if [ ! -f docs/notes/research_log.md ]; then
    echo "Creating research log..."
    cat > docs/notes/research_log.md << 'EOF'
# Research Log

## $(date +%Y-%m-%d) - Project Initialization

- Set up repository structure
- Created initial implementation scaffolding
- Configured environment and dependencies

### Next Steps
- [ ] Download and test base VLM models
- [ ] Implement PTQ baseline
- [ ] Set up evaluation pipeline

---
EOF
fi

echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run PTQ experiments:"
echo "  python scripts/run_ptq.py --config configs/ptq_gptq_4bit.yaml"
echo ""
echo "To run QAT experiments:"
echo "  python scripts/run_qat.py --config configs/qat_4bit.yaml"
