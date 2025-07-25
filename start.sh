#!/bin/bash

# Set environment variables to fix graphics issues
export LIBGL_ALWAYS_SOFTWARE=1
export ELECTRON_DISABLE_GPU=1
export DISPLAY=:0

# Activate virtual environment
source venv/bin/activate

# Kill any existing processes on port 3001
pkill -f "python.*backend_server.py" 2>/dev/null || true
pkill -f "electron" 2>/dev/null || true
sleep 2

# Start the app
echo "🐘 Starting Elephant ID Desktop App..."
echo "📡 Backend will start automatically"
echo "🌐 If Electron window doesn't open, try: http://localhost:3001"
echo ""

npm run dev
