#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install -r requirements.txt