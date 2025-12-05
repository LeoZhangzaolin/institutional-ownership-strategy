#!/bin/bash
# Setup directory structure for institutional ownership strategy

echo "=========================================="
echo "Institutional Ownership Strategy Setup"
echo "=========================================="
echo ""

# Create directories
echo "[1/5] Creating directory structure..."
mkdir -p data/{13f_parquet,cache,models,signals}
mkdir -p logs
mkdir -p reports
mkdir -p config
mkdir -p docs
mkdir -p tests

# Create .gitkeep files
touch data/.gitkeep
touch logs/.gitkeep
touch reports/.gitkeep

# Create __init__.py
touch src/__init__.py
touch tests/__init__.py

echo "✓ Directories created"

# Copy example config
echo ""
echo "[2/5] Setting up configuration..."
if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/config.example.yaml" ]; then
        cp config/config.example.yaml config/config.yaml
        echo "✓ Created config/config.yaml from example"
        echo "  ⚠️  EDIT config/config.yaml with your settings!"
    else
        echo "⚠️  config.example.yaml not found"
    fi
else
    echo "✓ config/config.yaml already exists"
fi

# Check Python
echo ""
echo "[3/5] Checking Python..."
if command -v python3 &> /dev/null; then
    python3 --version
    echo "✓ Python found"
else
    echo "❌ Python not found - please install Python 3.9+"
    exit 1
fi

# Check if virtual environment exists
echo ""
echo "[4/5] Checking virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate instructions
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo "   # On Windows: venv\\Scripts\\activate"
echo ""
echo "2. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Configure your settings:"
echo "   nano config/config.yaml"
echo "   # Add WRDS username, broker API keys"
echo ""
echo "4. Add your 13F data:"
echo "   # Copy your 13F parquet files to data/13f_parquet/"
echo "   # Expected format: yq=YYYY-QN/data.parquet"
echo ""
echo "5. Test installation:"
echo "   python -c 'from src.utils import load_config; print(load_config())'"
echo ""
echo "6. Build data:"
echo "   python scripts/quarterly_update.py update-data"
echo ""
echo "=========================================="
echo "Ready to deploy! 🚀"
echo "=========================================="