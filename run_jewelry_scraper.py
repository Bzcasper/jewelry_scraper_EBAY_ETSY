#!/usr/bin/env python3
"""
Simple Jewelry Scraper Runner
============================

Direct script to run the jewelry scraping system without complex imports.
Based on successful end-to-end tests from the spawned agents.
"""

import os
import sys
import time
import asyncio
import sqlite3
import yaml
from pathlib import Path
from datetime import datetime
import json

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def load_config():
    """Load jewelry scraping configuration"""
    config_path = Path("config/jewelry_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {
            'scraping': {
                'categories': ['rings', 'necklaces', 'earrings', 'bracelets', 'watches', 'pendants'],
                'rate_limit': 2.0,
                'max_retries': 3,
                'timeout': 30
            },
            'database': {
                'path': 'data/jewelry_scraping.db'
            },
            'images': {
                'download_path': 'storage/images/',
                'max_size': '1920x1080',
                'quality': 85
            }
        }

def init_database(db_path):
    """Initialize SQLite database"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS listings (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                price REAL,
                category TEXT,
                listing_url TEXT UNIQUE,
                image_urls TEXT,
                description TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print(f"✓ Database initialized: {db_path}")

async def scrape_jewelry_simple():
    """Simple jewelry scraping function using crawl4ai"""
    from crawl4ai import AsyncWebCrawler
    
    # Load config
    config = load_config()
    
    # Initialize database
    db_path = config['database']['path']
    init_database(db_path)
    
    # Create storage directories
    storage_path = Path(config['images']['download_path'])
    storage_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Jewelry Scraping Operation")
    print(f"Database: {db_path}")
    print(f"Storage: {storage_path}")
    
    # Keywords to scrape
    keywords = config['scraping']['categories'][:2]  # Limit to first 2 categories for demo
    
    async with AsyncWebCrawler(
        headless=True,
        browser_type="chromium",
        verbose=True
    ) as crawler:
        
        for keyword in keywords:
            print(f"\n🔍 Scraping: {keyword}")
            
            # eBay search URL
            search_url = f"https://www.ebay.com/sch/i.html?_nkw={keyword}+jewelry&_sacat=0&_udlo=10&_udhi=500"
            
            try:
                # Crawl the search page
                result = await crawler.arun(
                    url=search_url,
                    word_count_threshold=10,
                    extraction_strategy="cosine_similarity",
                    chunking_strategy="by_title"
                )
                
                if result.success:
                    print(f"✓ Successfully scraped {keyword} page")
                    print(f"  Content length: {len(result.markdown)} characters")
                    
                    # Extract basic info (simplified)
                    listings_found = result.markdown.count("$")  # Count price indicators
                    
                    # Store in database (simplified for demo)
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO listings 
                            (id, title, category, listing_url, description, scraped_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            f"{keyword}_{int(time.time())}",
                            f"{keyword.title()} Jewelry Search Results",
                            keyword,
                            search_url,
                            f"Found {listings_found} potential listings",
                            datetime.now()
                        ))
                        conn.commit()
                    
                    print(f"  Stored data for {keyword}")
                    
                else:
                    print(f"✗ Failed to scrape {keyword}: {result.error_message}")
                    
            except Exception as e:
                print(f"✗ Error scraping {keyword}: {e}")
            
            # Rate limiting
            await asyncio.sleep(config['scraping']['rate_limit'])
    
    print("\n🎉 Scraping completed!")
    
    # Show results
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM listings")
        total_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT category, COUNT(*) FROM listings GROUP BY category")
        by_category = dict(cursor.fetchall())
    
    print(f"📊 Results: {total_count} total listings")
    for category, count in by_category.items():
        print(f"  - {category}: {count}")

def run_cli_command(command):
    """Run a CLI command"""
    if command == "status":
        config = load_config()
        db_path = Path(config['database']['path'])
        
        print("🔍 System Status Check")
        print(f"Config file: {'✓' if Path('config/jewelry_config.yaml').exists() else '✗'}")
        print(f"Database: {'✓' if db_path.exists() else '✗'} ({db_path})")
        print(f"Storage: {'✓' if Path(config['images']['download_path']).exists() else '✗'}")
        
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM listings")
                count = cursor.fetchone()[0]
                print(f"Listings in database: {count}")
        
    elif command == "scrape":
        print("Starting jewelry scraping...")
        asyncio.run(scrape_jewelry_simple())
        
    elif command == "setup":
        print("🔧 Setting up jewelry scraper...")
        config = load_config()
        
        # Create directories
        Path("config").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("storage/images").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Save config
        with open("config/jewelry_config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Initialize database
        init_database(config['database']['path'])
        
        print("✅ Setup completed!")
        print("Next steps:")
        print("  python run_jewelry_scraper.py scrape  # Start scraping")
        print("  python run_jewelry_scraper.py status  # Check status")
        
    else:
        print("Available commands:")
        print("  setup   - Initial setup")
        print("  scrape  - Start scraping jewelry")
        print("  status  - Show system status")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        run_cli_command(command)
    else:
        run_cli_command("status")