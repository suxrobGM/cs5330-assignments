#!/bin/bash

# Navigate to the parent directory
cd ..

# Remove the .venv directory if it exists
if [ -d ".venv" ]; then
    echo "Removing .venv directory..."
    rm -rf .venv
fi

# Find and remove all __pycache__ directories
echo "Removing all __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup complete."
