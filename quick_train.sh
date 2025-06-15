#!/bin/bash

echo "🚀 Quick Job Fraud Detection Training"
echo "======================================"

# Check if training data exists
if [ ! -f "training_data.csv" ]; then
    echo "❌ training_data.csv not found!"
    echo "Please make sure the file exists in the project root."
    exit 1
fi

# Show dataset info
echo "📊 Dataset Info:"
wc -l training_data.csv
echo ""

# Ask user for model type
echo "Choose training speed:"
echo "1. FAST (2-4 minutes, 85-90% accuracy)"
echo "2. ENHANCED (5-8 minutes, 90-94% accuracy)"
echo "3. ULTRA SMALL (10-15 minutes, 94-98% accuracy, smaller dataset)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Training FAST model..."
        python scripts/train_enhanced_model_fast.py
        ;;
    2)
        echo "🚀 Training ENHANCED model..."
        python scripts/train_enhanced_model.py
        ;;
    3)
        echo "🚀 Creating smaller dataset and training ULTRA model..."
        head -5001 training_data.csv > training_data_small.csv
        echo "📊 Using 5000 rows for ultra training..."
        python scripts/train_ultra_enhanced_model.py
        ;;
    *)
        echo "❌ Invalid choice. Using FAST model..."
        python scripts/train_enhanced_model_fast.py
        ;;
esac

echo ""
echo "✅ Training completed!"
echo "🔍 Test your model with:"
echo "python scripts/predict_enhanced.py test_jobs.csv results.csv"
