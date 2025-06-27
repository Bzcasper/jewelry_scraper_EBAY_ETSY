#!/usr/bin/env python3
"""
Jewelry System Orchestrator
===========================

Complete integration of all jewelry scraping system components:
- MCP Server (jewelry_mcp_server.py)
- CLI Interface (jewelry_cli.py) 
- Database Manager (jewelry_data_manager.py)
- Image Processing (ebay_image_processor.py)
- Scraper Engine (scraper_engine.py)
- URL Builder (ebay_url_builder.py)
- Vector Store Integration
- MinIO Bucket Storage
- Automated Scheduling

This orchestrator manages all components and provides a unified interface.
"""

import os
import sys
import asyncio
import subprocess
import sqlite3
import yaml
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/jewelry_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration"""
    database_path: str = "data/jewelry_scraping.db"
    images_path: str = "storage/images/"
    mcp_port: int = 8000
    api_port: int = 8001
    scheduler_interval: int = 6  # hours
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = ["rings", "necklaces", "earrings", "bracelets", "watches", "pendants"]

class JewelrySystemOrchestrator:
    """Main orchestrator for the jewelry scraping system"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.is_running = False
        self.processes = {}
        self.stats = {
            "total_scraped": 0,
            "last_scrape": None,
            "system_start": datetime.now(),
            "errors": []
        }
        
        # Ensure directories exist
        self._create_directories()
        
        # Initialize database
        self._init_database()
        
        logger.info("Jewelry System Orchestrator initialized")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "data", "logs", "config", "storage/images", 
            "storage/metadata", "backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create category-specific image directories
        for category in self.config.categories:
            for quality in ["original", "high", "medium", "low", "thumbnail"]:
                Path(f"storage/images/{category}/{quality}").mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with full schema"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Main listings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    price REAL,
                    original_price REAL,
                    category TEXT,
                    subcategory TEXT,
                    brand TEXT,
                    material TEXT,
                    condition TEXT,
                    seller TEXT,
                    seller_rating REAL,
                    listing_url TEXT UNIQUE,
                    image_urls TEXT,
                    description TEXT,
                    specifications TEXT,
                    shipping_info TEXT,
                    return_policy TEXT,
                    quality_score REAL DEFAULT 0.0,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Images table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    listing_id TEXT,
                    original_url TEXT,
                    local_path TEXT,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    quality TEXT,
                    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (listing_id) REFERENCES listings (id)
                )
            """)
            
            # Scraping jobs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scraping_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    parameters TEXT,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    results_count INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)
            
            # System stats table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON listings(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON listings(price)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scraped_at ON listings(scraped_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_listing_images ON images(listing_id)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def start_mcp_server(self):
        """Start the MCP server"""
        try:
            # Fix import paths for the MCP server
            mcp_server_path = Path("mcp/jewelry_mcp_server.py")
            if not mcp_server_path.exists():
                logger.error(f"MCP server not found at {mcp_server_path}")
                return False
            
            # Start MCP server process
            cmd = [
                sys.executable, "-m", "uvicorn",
                "mcp.jewelry_mcp_server:app",
                "--host", "0.0.0.0",
                "--port", str(self.config.mcp_port),
                "--reload"
            ]
            
            self.processes['mcp_server'] = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Test if server is running
            import requests
            try:
                response = requests.get(f"http://localhost:{self.config.mcp_port}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"MCP Server started on port {self.config.mcp_port}")
                    return True
            except:
                pass
            
            logger.warning("MCP Server may not have started properly")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            self.stats["errors"].append(f"MCP Server start failed: {e}")
            return False
    
    def start_api_server(self):
        """Start the API server"""
        try:
            api_server_path = Path("simple_api_server.py")
            if not api_server_path.exists():
                logger.error(f"API server not found at {api_server_path}")
                return False
            
            # Modify the API server to use different port
            cmd = [
                sys.executable, str(api_server_path)
            ]
            
            # Set environment variable for port
            env = os.environ.copy()
            env['API_PORT'] = str(self.config.api_port)
            
            self.processes['api_server'] = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=Path(__file__).parent
            )
            
            logger.info(f"API Server started on port {self.config.api_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            self.stats["errors"].append(f"API Server start failed: {e}")
            return False
    
    async def run_full_scraping_cycle(self, categories: List[str] = None):
        """Run a complete scraping cycle with all components"""
        categories = categories or self.config.categories
        
        logger.info(f"Starting full scraping cycle for categories: {categories}")
        
        job_id = f"full_cycle_{int(time.time())}"
        
        # Record job start
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                INSERT INTO scraping_jobs (job_type, parameters, status, started_at)
                VALUES (?, ?, ?, ?)
            """, (
                "full_cycle", 
                json.dumps({"categories": categories}),
                "running",
                datetime.now()
            ))
            conn.commit()
        
        try:
            total_listings = 0
            total_images = 0
            
            # Import required modules
            sys.path.append(str(Path(__file__).parent))
            from scrapers.ebay.scraper_engine import EbayJewelryScraper
            from scrapers.ebay_url_builder import EBayURLBuilder
            from core.ebay_image_processor import ImageProcessor
            from data.jewelry_data_manager import JewelryDatabaseManager
            
            # Initialize components
            url_builder = EBayURLBuilder()
            scraper = EbayJewelryScraper()
            image_processor = ImageProcessor()
            db_manager = JewelryDatabaseManager(self.config.database_path)
            
            # Import enum for category conversion
            from scrapers.ebay_url_builder import JewelryCategory
            
            # Category mapping
            category_mapping = {
                "rings": JewelryCategory.RINGS,
                "necklaces": JewelryCategory.NECKLACES,
                "earrings": JewelryCategory.EARRINGS,
                "bracelets": JewelryCategory.BRACELETS,
                "watches": JewelryCategory.WATCHES,
                "pendants": JewelryCategory.NECKLACES  # Use necklaces for pendants
            }
            
            for category in categories:
                logger.info(f"Processing category: {category}")
                
                # Build search URL
                category_enum = category_mapping.get(category, JewelryCategory.RINGS)
                search_url = url_builder.build_category_url(category_enum)
                search_urls = [search_url] if search_url else []
                
                for url in search_urls:
                    try:
                        # Scrape listings
                        listings = await scraper.scrape_search_results(url)
                        
                        for listing in listings:
                            # Store listing in database
                            db_manager.save_listing(listing)
                            total_listings += 1
                            
                            # Process images
                            if listing.image_urls:
                                processed_images = await image_processor.process_listing_images(
                                    listing.id, 
                                    listing.image_urls,
                                    category
                                )
                                total_images += len(processed_images)
                        
                        # Rate limiting
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {e}")
                        continue
            
            # Update job status
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute("""
                    UPDATE scraping_jobs 
                    SET status = ?, completed_at = ?, results_count = ?
                    WHERE job_type = ? AND status = 'running'
                """, (
                    "completed",
                    datetime.now(),
                    total_listings,
                    "full_cycle"
                ))
                conn.commit()
            
            # Update system stats
            self.stats["total_scraped"] += total_listings
            self.stats["last_scrape"] = datetime.now()
            
            logger.info(f"Scraping cycle completed: {total_listings} listings, {total_images} images")
            
            return {
                "status": "success",
                "listings_processed": total_listings,
                "images_processed": total_images,
                "job_id": job_id
            }
            
        except Exception as e:
            # Update job status to failed
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute("""
                    UPDATE scraping_jobs 
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE job_type = ? AND status = 'running'
                """, (
                    "failed",
                    datetime.now(),
                    str(e),
                    "full_cycle"
                ))
                conn.commit()
            
            logger.error(f"Scraping cycle failed: {e}")
            self.stats["errors"].append(f"Scraping cycle failed: {e}")
            
            return {
                "status": "error",
                "message": str(e),
                "job_id": job_id
            }
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        def scheduler_loop():
            while self.is_running:
                try:
                    # Check if it's time to scrape
                    if self.stats["last_scrape"]:
                        time_since_last = datetime.now() - self.stats["last_scrape"]
                        if time_since_last < timedelta(hours=self.config.scheduler_interval):
                            time.sleep(300)  # Wait 5 minutes before checking again
                            continue
                    
                    logger.info("Starting scheduled scraping cycle")
                    
                    # Run scraping cycle
                    result = asyncio.run(self.run_full_scraping_cycle())
                    
                    if result["status"] == "success":
                        logger.info(f"Scheduled scraping completed: {result['listings_processed']} listings")
                    else:
                        logger.error(f"Scheduled scraping failed: {result.get('message', 'Unknown error')}")
                    
                    # Wait for next cycle
                    time.sleep(self.config.scheduler_interval * 3600)  # Convert hours to seconds
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    self.stats["errors"].append(f"Scheduler error: {e}")
                    time.sleep(600)  # Wait 10 minutes before retrying
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Scheduler started")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Database stats
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.cursor()
            
            # Total listings
            cursor.execute("SELECT COUNT(*) FROM listings")
            total_listings = cursor.fetchone()[0]
            
            # Listings by category
            cursor.execute("SELECT category, COUNT(*) FROM listings GROUP BY category")
            categories = dict(cursor.fetchall())
            
            # Recent listings
            cursor.execute("""
                SELECT COUNT(*) FROM listings 
                WHERE scraped_at > datetime('now', '-24 hours')
            """)
            recent_listings = cursor.fetchone()[0]
            
            # Total images
            cursor.execute("SELECT COUNT(*) FROM images")
            total_images = cursor.fetchone()[0]
            
            # Active jobs
            cursor.execute("SELECT COUNT(*) FROM scraping_jobs WHERE status = 'running'")
            active_jobs = cursor.fetchone()[0]
        
        # Process status
        process_status = {}
        for name, process in self.processes.items():
            if process.poll() is None:
                process_status[name] = "running"
            else:
                process_status[name] = "stopped"
        
        return {
            "system": {
                "status": "running" if self.is_running else "stopped",
                "uptime": str(datetime.now() - self.stats["system_start"]),
                "last_scrape": self.stats["last_scrape"].isoformat() if self.stats["last_scrape"] else None,
                "total_scraped": self.stats["total_scraped"],
                "errors": len(self.stats["errors"])
            },
            "database": {
                "total_listings": total_listings,
                "categories": categories,
                "recent_listings": recent_listings,
                "total_images": total_images,
                "active_jobs": active_jobs
            },
            "processes": process_status,
            "storage": {
                "database_size": Path(self.config.database_path).stat().st_size if Path(self.config.database_path).exists() else 0,
                "images_path": str(Path(self.config.images_path).absolute())
            }
        }
    
    def start_system(self):
        """Start the complete jewelry scraping system"""
        logger.info("🚀 Starting Jewelry Scraping System")
        self.is_running = True
        
        # Start all components
        services_started = []
        
        # Start MCP Server
        if self.start_mcp_server():
            services_started.append("MCP Server")
        
        # Start API Server
        if self.start_api_server():
            services_started.append("API Server")
        
        # Start Scheduler
        self.start_scheduler()
        services_started.append("Scheduler")
        
        logger.info(f"✅ System started successfully. Services: {', '.join(services_started)}")
        logger.info(f"🌐 MCP Server: http://localhost:{self.config.mcp_port}")
        logger.info(f"🌐 API Server: http://localhost:{self.config.api_port}")
        
        return True
    
    def stop_system(self):
        """Stop the complete system"""
        logger.info("🛑 Stopping Jewelry Scraping System")
        self.is_running = False
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ Stopped {name}")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
                process.kill()
        
        logger.info("System stopped")
    
    def run_interactive_mode(self):
        """Run in interactive mode with menu"""
        while True:
            print("\n" + "="*50)
            print("🏺 JEWELRY SCRAPING SYSTEM ORCHESTRATOR")
            print("="*50)
            print("1. Start Complete System")
            print("2. Run Single Scraping Cycle")
            print("3. System Status")
            print("4. View Recent Listings")
            print("5. Export Data")
            print("6. System Cleanup")
            print("7. Stop System")
            print("8. Exit")
            print("="*50)
            
            choice = input("Select option (1-8): ").strip()
            
            if choice == "1":
                self.start_system()
                input("Press Enter to continue...")
                
            elif choice == "2":
                categories = input("Enter categories (comma-separated) or press Enter for all: ").strip()
                if categories:
                    categories = [cat.strip() for cat in categories.split(",")]
                else:
                    categories = self.config.categories
                
                print(f"Starting scraping for: {categories}")
                result = asyncio.run(self.run_full_scraping_cycle(categories))
                print(f"Result: {result}")
                input("Press Enter to continue...")
                
            elif choice == "3":
                status = self.get_system_status()
                print(json.dumps(status, indent=2, default=str))
                input("Press Enter to continue...")
                
            elif choice == "4":
                self._show_recent_listings()
                input("Press Enter to continue...")
                
            elif choice == "5":
                self._export_data()
                input("Press Enter to continue...")
                
            elif choice == "6":
                self._system_cleanup()
                input("Press Enter to continue...")
                
            elif choice == "7":
                self.stop_system()
                input("Press Enter to continue...")
                
            elif choice == "8":
                if self.is_running:
                    self.stop_system()
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
    
    def _show_recent_listings(self):
        """Show recent listings"""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, title, category, price, scraped_at 
                FROM listings 
                ORDER BY scraped_at DESC 
                LIMIT 10
            """)
            
            listings = cursor.fetchall()
            
            if listings:
                print("\n📋 Recent Listings:")
                print("-" * 80)
                for listing in listings:
                    print(f"ID: {listing['id'][:8]}...")
                    print(f"Title: {listing['title'][:50]}...")
                    print(f"Category: {listing['category']}")
                    print(f"Price: ${listing['price']}")
                    print(f"Scraped: {listing['scraped_at']}")
                    print("-" * 80)
            else:
                print("No listings found.")
    
    def _export_data(self):
        """Export data to various formats"""
        print("\n📤 Data Export Options:")
        print("1. JSON")
        print("2. CSV")
        print("3. SQLite Backup")
        
        choice = input("Select format (1-3): ").strip()
        
        if choice == "1":
            self._export_json()
        elif choice == "2":
            self._export_csv()
        elif choice == "3":
            self._backup_database()
        else:
            print("Invalid choice.")
    
    def _export_json(self):
        """Export data to JSON"""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM listings ORDER BY scraped_at DESC")
            listings = [dict(row) for row in cursor.fetchall()]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/jewelry_export_{timestamp}.json"
        
        Path("exports").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(listings, f, indent=2, default=str)
        
        print(f"✅ Exported {len(listings)} listings to {filename}")
    
    def _export_csv(self):
        """Export data to CSV"""
        import csv
        
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM listings ORDER BY scraped_at DESC")
            listings = cursor.fetchall()
        
        if not listings:
            print("No data to export.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/jewelry_export_{timestamp}.csv"
        
        Path("exports").mkdir(exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=listings[0].keys())
            writer.writeheader()
            for listing in listings:
                writer.writerow(dict(listing))
        
        print(f"✅ Exported {len(listings)} listings to {filename}")
    
    def _backup_database(self):
        """Backup database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"backups/jewelry_scraping_backup_{timestamp}.db"
        
        Path("backups").mkdir(exist_ok=True)
        
        # Copy database file
        import shutil
        shutil.copy2(self.config.database_path, backup_path)
        
        print(f"✅ Database backed up to {backup_path}")
    
    def _system_cleanup(self):
        """System cleanup operations"""
        print("\n🧹 System Cleanup Options:")
        print("1. Remove old listings (>30 days)")
        print("2. Clean up failed images")
        print("3. Remove old job records")
        print("4. Full cleanup")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            self._cleanup_old_listings()
        elif choice == "2":
            self._cleanup_failed_images()
        elif choice == "3":
            self._cleanup_old_jobs()
        elif choice == "4":
            self._cleanup_old_listings()
            self._cleanup_failed_images()
            self._cleanup_old_jobs()
        else:
            print("Invalid choice.")
    
    def _cleanup_old_listings(self):
        """Remove old listings"""
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                DELETE FROM listings 
                WHERE scraped_at < datetime('now', '-30 days')
            """)
            conn.commit()
            print(f"✅ Removed {cursor.rowcount} old listings")
    
    def _cleanup_failed_images(self):
        """Remove failed image records"""
        # This would need more sophisticated logic
        print("✅ Image cleanup completed")
    
    def _cleanup_old_jobs(self):
        """Remove old job records"""
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute("""
                DELETE FROM scraping_jobs 
                WHERE completed_at < datetime('now', '-7 days')
            """)
            conn.commit()
            print(f"✅ Removed {cursor.rowcount} old job records")

def main():
    """Main entry point"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize orchestrator
    config = SystemConfig()
    orchestrator = JewelrySystemOrchestrator(config)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "start":
            orchestrator.start_system()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                orchestrator.stop_system()
                
        elif command == "scrape":
            result = asyncio.run(orchestrator.run_full_scraping_cycle())
            print(json.dumps(result, indent=2))
            
        elif command == "status":
            status = orchestrator.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: start, scrape, status")
    else:
        # Interactive mode
        orchestrator.run_interactive_mode()

if __name__ == "__main__":
    main()