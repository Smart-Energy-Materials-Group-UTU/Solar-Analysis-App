#!/bin/bash

# Check if virtual environment exists
if [ ! -d "thesis" ]; then
    echo "Virtual environment not found, setting it up now..."
    python3 -m venv "thesis"
    echo "Installing dependencies..."
    "thesis/bin/pip" install -r "requirements.txt"
fi

# Activate the virtual environment
source "thesis/bin/activate"

# Run the Python application
echo "Launching the Solar Analysis Application..."
python3 "Project Code/main_algorithm.py"

# Deactivate the virtual environment when finished
deactivate