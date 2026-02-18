#!/bin/bash

# Jarvis Web Dashboard Launcher
# Starts the web-based interface for GPU trace analysis

echo "=========================================="
echo "  🤖 Jarvis Web Dashboard Launcher"
echo "=========================================="
echo ""

# Check if in the correct directory
if [ ! -f "jarvis_web_dashboard.py" ]; then
    echo "❌ Error: jarvis_web_dashboard.py not found"
    echo "Please run this script from the TraceLens-Jarvis directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Activate virtual environment
echo "✓ Activating virtual environment (gpu_jarvis_analysis)..."
source gpu_jarvis_analysis/bin/activate

# Install web dependencies
echo "✓ Installing web dependencies..."
pip install -q flask werkzeug markdown

# Check if main dependencies are installed
if ! python -c "import sys; sys.path.insert(0, 'Analysis'); from jarvis_analysis import JarvisAnalyzer" 2>/dev/null; then
    echo "⚠️  Installing Jarvis analysis dependencies..."
    pip install -q -r requirements.txt 2>/dev/null || true
fi

# Create necessary directories
mkdir -p uploads web_reports templates static/css static/js

# Set environment variables
export FLASK_APP=jarvis_web_dashboard.py
export FLASK_ENV=development

# Display startup info
echo ""
echo "=========================================="
echo "  🚀 Starting Jarvis Web Dashboard"
echo "=========================================="
echo ""
echo "📊 Dashboard URL: http://localhost:5000"
echo "📁 Uploads folder: ./uploads"
echo "📁 Reports folder: ./web_reports"
echo ""
echo "Features:"
echo "  • Upload and analyze GPU traces"
echo "  • Compare performance between GPUs"
echo "  • View interactive reports and plots"
echo "  • Track analysis job progress"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Start the Flask app
python3 jarvis_web_dashboard.py
