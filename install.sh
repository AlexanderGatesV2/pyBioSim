#!/bin/bash

# Create a virtual environment (optional)
echo "Do you want to create a virtual environment? (y/n)"
read create_venv

if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    
    # Activate virtual environment
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
    elif [ -f venv/Scripts/activate ]; then
        source venv/Scripts/activate
    else
        echo "Failed to activate virtual environment. Please activate it manually."
        exit 1
    fi
    
    echo "Virtual environment created and activated."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing pyBioSim in development mode..."
pip install -e .

echo "Installation complete!"
echo "Run 'python main.py' to start the simulation."
