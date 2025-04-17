#!/bin/bash
echo "LUMINA V7.5 - Dependencies Installation"
echo "======================================="
echo ""

# Check for Python
python3 --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 not found. Please install Python 3.8 or newer."
    echo "Visit https://www.python.org/downloads/"
    exit 1
fi

echo "Python found. Installing required packages..."
echo ""

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt file..."
    cat > requirements.txt << EOF
PySide6>=6.4.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
pyqtgraph>=0.12.0
python-dotenv>=0.19.0
requests>=2.26.0
SQLAlchemy>=1.4.0
scipy>=1.7.0
scikit-learn>=1.0.0
tqdm>=4.62.0
pillow>=8.3.0
psutil>=5.8.0
pytest>=6.2.5
beautifulsoup4>=4.9.0
EOF
fi

# Install packages
echo "Installing packages from requirements.txt..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install some packages."
    echo "You may need to run with sudo or use:"
    echo "    pip3 install --user -r requirements.txt"
    echo ""
    exit 1
fi

echo ""
echo "All dependencies successfully installed!"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/neural
mkdir -p data/memory
mkdir -p data/onsite_memory
mkdir -p data/seed
mkdir -p data/dream
mkdir -p data/autowiki
mkdir -p data/consciousness
mkdir -p data/breath
mkdir -p data/v7.5
mkdir -p data/conversations
mkdir -p data/db
mkdir -p logs/db
mkdir -p logs/chat
mkdir -p logs/monitor

echo ""
echo "Setup complete. You can now run the LUMINA V7.5 system by executing:"
echo "    bash run_v7_holographic.sh"
echo "" 