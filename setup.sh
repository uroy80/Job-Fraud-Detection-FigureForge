#!/bin/bash
set -e

echo "🚀 Job Fraud Detection - Complete Setup"
echo "======================================"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn nltk tqdm scikit-optimize uuid

# Alternative for some systems
# pip3 install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn nltk tqdm scikit-optimize uuid

# Download NLTK data
echo "📚 Downloading NLTK data..."
python scripts/download_nltk_data.py

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p tmp
mkdir -p uploads
mkdir -p results

echo ""
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Add your training_data.csv file to the project root"
echo "2. Run: npm run dev (for web interface)"
echo "3. Or train model: python scripts/train_ultra_enhanced_model.py"
echo ""
