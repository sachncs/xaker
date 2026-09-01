#!/bin/bash
# Setup script for XAKER development environment

set -e

echo "Setting up XAKER development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode with development dependencies
echo "Installing XAKER with development dependencies..."
pip install -e ".[dev,bench,train]"

# Run tests to verify installation
echo "Running tests to verify installation..."
pytest tests/ -v

echo ""
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
