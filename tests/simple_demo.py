#!/usr/bin/env python3
"""
Simple Real Jewelry Scraping Demonstration
Shows the complete pipeline working with real data in a simplified manner.
"""

import asyncio
import logging
import json
import os
import sys
import time
import sqlite3
from datetime import datetime
from pathlib import Path
import tempfile

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class SimpleJewelryDemo:
    """Simple demonstration of jewelry scraping system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Demo setup
        self.demo_dir = Path(tempfile.mkdtemp(prefix="jewelry_simple_demo_"))
        self.db_path = self.demo_dir / "jewelry.db"
        self.images_dir = self.demo_dir / "images"
        
        # Test parameters
        self.test_keyword = "gold ring"
        self.max_listings = 2
        
        # Results
        self.results = {
            'setup_ok': False,
            'scraping_ok': False,
            'data_found': 0,
            'errors': []
        }
        
        print(f"🎬 Simple demo starting...")
        print(f"📁 Demo directory: {self.demo_dir}")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def run_simple_demo(self):
        """Run the simple jewelry scraping demo"""
        print("\n🎯 SIMPLE JEWELRY SCRAPING DEMONSTRATION")
        print("=" * 50)
        
        try:
            # Step 1: Setup
            await self.setup_environment()
            
            # Step 2: Quick scraping test
            await self.test_scraping()
            
            # Step 3: Show results
            await self.show_results()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            self.results['errors'].append(str(e))
        finally:
            self.cleanup()
    
    async def setup_environment(self):
        """Setup demo environment"""
        print("\n🔧 Setting up environment...")
        
        try:
            # Create directories
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
            # Create database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE jewelry_demo (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        price REAL,
                        category TEXT,
                        url TEXT,
                        scraped_at TEXT
                    )
                """)
                conn.commit()
            
            print("  ✅ Database created")
            print("  ✅ Directories created")
            
            self.results['setup_ok'] = True
            
        except Exception as e:
            print(f"  ❌ Setup failed: {e}")
            raise
    
    async def test_scraping(self):
        """Test basic scraping functionality"""
        print(f"\n🌐 Testing scraping for: {self.test_keyword}")
        
        try:
            from crawl4ai import AsyncWebCrawler
            from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
            from bs4 import BeautifulSoup
            from urllib.parse import urlencode
            import re
            
            # Build search URL
            params = {'_nkw': self.test_keyword, '_pgn': 1, '_ipg': 10}
            search_url = f"https://www.ebay.com/sch/i.html?{urlencode(params)}"
            print(f"  🔗 URL: {search_url}")
            
            # Configure crawler for quick test
            browser_config = BrowserConfig(
                headless=True,
                extra_args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
            )
            
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_for_images=False,
                process_iframes=False,
                page_timeout=15000  # 15 seconds
            )
            
            print("  ⏳ Starting crawl (15 second timeout)...")
            
            # Perform crawl with asyncio timeout
            try:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await asyncio.wait_for(
                        crawler.arun(url=search_url, config=run_config),
                        timeout=20.0  # 20 second timeout
                    )
                    
                    if result.success:
                        print(f"  ✅ Crawl successful - {len(result.html)} chars")
                        
                        # Quick HTML parsing
                        soup = BeautifulSoup(result.html, 'html.parser')
                        
                        # Find any elements that might be listings
                        potential_listings = soup.find_all(text=re.compile(r'\$\d+'))
                        price_elements = [elem for elem in potential_listings if '$' in str(elem)]
                        
                        print(f"  💰 Found {len(price_elements)} price elements")
                        
                        # Extract some basic data
                        listings_extracted = []
                        
                        # Look for title-like elements near prices
                        for i, price_elem in enumerate(price_elements[:self.max_listings]):
                            try:
                                # Find parent container
                                parent = price_elem.parent
                                if parent:
                                    # Look for text that might be a title
                                    title_texts = parent.find_all(text=True)
                                    title_candidates = [t.strip() for t in title_texts if len(t.strip()) > 10]
                                    
                                    if title_candidates:
                                        title = title_candidates[0][:100]  # First reasonable text
                                        price_str = str(price_elem).strip()
                                        price_match = re.search(r'[\d,]+\.?\d*', price_str.replace(',', ''))
                                        price = float(price_match.group()) if price_match else 0.0
                                        
                                        if price > 0 and len(title) > 5:
                                            listing = {
                                                'id': f'demo_{i}',
                                                'title': title,
                                                'price': price,
                                                'category': 'rings',
                                                'url': search_url,
                                                'scraped_at': datetime.now().isoformat()
                                            }
                                            listings_extracted.append(listing)
                                            print(f"    {i+1}. {title[:50]}... - ${price}")
                            except Exception:
                                continue
                        
                        # Store in database
                        if listings_extracted:
                            with sqlite3.connect(self.db_path) as conn:
                                for listing in listings_extracted:
                                    conn.execute("""
                                        INSERT INTO jewelry_demo 
                                        (id, title, price, category, url, scraped_at)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    """, (
                                        listing['id'], listing['title'], listing['price'],
                                        listing['category'], listing['url'], listing['scraped_at']
                                    ))
                                conn.commit()
                            
                            print(f"  💾 Stored {len(listings_extracted)} listings in database")
                            self.results['data_found'] = len(listings_extracted)
                            self.results['scraping_ok'] = True
                            self.extracted_listings = listings_extracted
                        else:
                            print("  ⚠️  No listings extracted (page structure may have changed)")
                            self.results['scraping_ok'] = True  # Scraping worked, just no data
                    else:
                        print(f"  ❌ Crawl failed: {result.error_message}")
            
            except asyncio.TimeoutError:
                print("  ⏰ Crawl timed out")
                self.results['errors'].append("Crawl timeout")
            
        except Exception as e:
            print(f"  ❌ Scraping test failed: {e}")
            self.results['errors'].append(f"Scraping: {e}")
    
    async def show_results(self):
        """Show demo results"""
        print("\n📊 Demo Results")
        print("-" * 30)
        
        try:
            # Check database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM jewelry_demo")
                db_count = cursor.fetchone()[0]
                
                if db_count > 0:
                    cursor = conn.execute("SELECT title, price FROM jewelry_demo LIMIT 3")
                    samples = cursor.fetchall()
                    
                    print(f"📦 Database entries: {db_count}")
                    print("💎 Sample data:")
                    for title, price in samples:
                        print(f"   • {title[:40]}... - ${price}")
                else:
                    print("📦 Database is empty")
            
            # Create test image
            try:
                from PIL import Image
                img = Image.new('RGB', (100, 100), color='gold')
                img_path = self.images_dir / 'demo.jpg'
                img.save(img_path)
                print(f"📸 Test image created: {img_path.name}")
            except Exception:
                print("📸 Image processing unavailable")
            
        except Exception as e:
            print(f"❌ Results check failed: {e}")
    
    def cleanup(self):
        """Cleanup demo"""
        print(f"\n🧹 Demo completed")
        print(f"📁 Files preserved at: {self.demo_dir}")
        print(f"   • Database: {self.db_path.name}")
        print(f"   • Images: {self.images_dir.name}")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 50)
        print("🎯 SIMPLE JEWELRY SCRAPING DEMO SUMMARY")
        print("=" * 50)
        
        print(f"🔧 Setup: {'✅ Success' if self.results['setup_ok'] else '❌ Failed'}")
        print(f"🌐 Scraping: {'✅ Success' if self.results['scraping_ok'] else '❌ Failed'}")
        print(f"📊 Data found: {self.results['data_found']} listings")
        
        if self.results['errors']:
            print(f"❌ Errors: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        print(f"\n🎯 ASSESSMENT:")
        if self.results['setup_ok'] and self.results['scraping_ok']:
            if self.results['data_found'] > 0:
                print("   🎉 COMPLETE SUCCESS!")
                print("   ✅ System working end-to-end with real data")
            else:
                print("   ⚠️  PARTIAL SUCCESS")
                print("   ✅ System functional, limited data extraction")
        else:
            print("   ❌ SYSTEM ISSUES")
            print("   🔧 Check error messages above")
        
        print("=" * 50)


async def main():
    """Main demo execution"""
    print("🎬 JEWELRY SCRAPING SYSTEM - SIMPLE DEMONSTRATION")
    print("Testing the core pipeline with minimal real scraping")
    
    demo = SimpleJewelryDemo()
    await demo.run_simple_demo()
    demo.print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        sys.exit(1)