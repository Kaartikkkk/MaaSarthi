#!/bin/bash
# 🚀 MaaSarthi - Optimized Server Launcher
# Start the high-performance Flask application

echo "========================================="
echo "🚀 Starting MaaSarthi Server"
echo "========================================="
echo ""

# Navigate to project directory
cd /Users/kartik/Documents/MaaSarthi

# Activate virtual environment
source .venv/bin/activate

# Check if port 5001 is in use
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 5001 is already in use"
    echo "Killing existing process..."
    lsof -ti:5001 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start the optimized server
echo "✅ Launching Flask server..."
echo "   - Debug mode: OFF"
echo "   - Lazy loading: ENABLED"  
echo "   - Dataset caching: ENABLED"
echo ""
echo "🌐 Server running at: http://localhost:5001"
echo "📊 Performance tests: python performance_test.py"
echo ""

/Users/kartik/Documents/MaaSarthi/.venv/bin/python app.py
