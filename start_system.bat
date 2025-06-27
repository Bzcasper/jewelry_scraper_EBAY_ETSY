@echo off
echo 🚀 Starting Jewelry Scraping System...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set environment variables
set PYTHONPATH=.
set JEWELRY_CONFIG_PATH=config\jewelry_config.yaml

REM Start the system orchestrator
python jewelry_system_orchestrator.py start
