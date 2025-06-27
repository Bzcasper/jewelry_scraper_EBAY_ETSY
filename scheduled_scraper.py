#!/usr/bin/env python3
"""
Scheduled Jewelry Scraper
=========================

Runs jewelry scraping on a schedule using keywords from YAML config.
Can be run via cron or as a daemon process.
"""

import time
import schedule
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

def load_config():
    """Load scraping configuration"""
    config_path = Path("config/jewelry_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {"scraping": {"categories": ["rings", "necklaces", "earrings"]}}

def run_scraper():
    """Run the jewelry scraper"""
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting scheduled scrape")
    
    try:
        scraper_path = Path(__file__).parent / "run_jewelry_scraper.py"
        result = subprocess.run([
            sys.executable, str(scraper_path), "scrape"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Scraping completed successfully")
        else:
            print(f"❌ Scraping failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Scraping timed out after 5 minutes")
    except Exception as e:
        print(f"🚨 Error running scraper: {e}")

def main():
    """Main scheduler function"""
    print("📅 Jewelry Scraper Scheduler Started")
    print("⏰ Will run every 6 hours")
    
    config = load_config()
    categories = config.get("scraping", {}).get("categories", ["rings"])
    print(f"🔍 Categories to scrape: {categories}")
    
    # Schedule scraping every 6 hours
    schedule.every(6).hours.do(run_scraper)
    
    # Also run once at startup
    run_scraper()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    if "--once" in sys.argv:
        # Run once and exit
        run_scraper()
    else:
        # Run as scheduler
        try:
            # Install schedule if needed
            import schedule
        except ImportError:
            print("Installing schedule package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "schedule"])
            import schedule
            
        main()