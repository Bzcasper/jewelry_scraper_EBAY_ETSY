#!/bin/bash
# Jewelry Scraping System Startup Script

echo "🚀 Starting Jewelry Scraping System..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=.
export JEWELRY_CONFIG_PATH=config/jewelry_config.yaml

# Start the system orchestrator
python jewelry_system_orchestrator.py start
