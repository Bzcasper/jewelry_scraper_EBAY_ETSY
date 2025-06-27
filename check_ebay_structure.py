#!/usr/bin/env python3
"""
Check actual HTML structure of working eBay search page
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

async def check_ebay_structure():
    """Check what the actual HTML structure looks like"""
    
    # Use a working URL
    test_url = "https://www.ebay.com/sch/i.html?_nkw=ring+jewelry"
    
    print(f"🔍 Analyzing: {test_url}")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(test_url)
        
        if not result.success:
            print(f"❌ Failed: {result.error_message}")
            return
            
        print(f"✅ Page fetched: {len(result.cleaned_html)} chars")
        
        # Parse HTML
        soup = BeautifulSoup(result.cleaned_html, 'html.parser')
        
        # Find listing-like elements
        print("\n🔍 Looking for listing patterns...")
        
        # Check for various possible selectors
        patterns = [
            'div[class*="item"]',
            'div[class*="listing"]', 
            'div[class*="result"]',
            'li[class*="item"]',
            'article',
            '[data-testid]',
            'div[class*="srp"]',
            'div[class*="s-"]',
            '.notranslate',
            '[class*="listingcard"]',
            '[class*="card"]'
        ]
        
        for pattern in patterns:
            elements = soup.select(pattern)
            if elements and len(elements) > 5:  # Only show patterns with multiple matches
                print(f"✅ {pattern:<25} | {len(elements):>3} found")
                
                # Show classes of first few elements
                classes = []
                for el in elements[:3]:
                    el_classes = el.get('class', [])
                    if el_classes:
                        classes.append(' '.join(el_classes))
                
                if classes:
                    print(f"   Classes: {classes[:2]}")
        
        # Look for specific eBay patterns
        print("\n🎯 eBay-specific patterns:")
        
        ebay_patterns = [
            ('Product links', 'a[href*="/itm/"]'),
            ('Price elements', '*[class*="price"]'),
            ('Title elements', '*[class*="title"]'),
            ('Image elements', 'img[src*="ebayimg"]'),
            ('Listing containers', 'div[class*="listing"]'),
            ('Card containers', 'div[class*="card"]')
        ]
        
        for name, pattern in ebay_patterns:
            elements = soup.select(pattern)
            print(f"{name:<20} | {pattern:<30} | {len(elements):>3} found")
            
            if elements and len(elements) > 0:
                # Show first element info
                first = elements[0]
                parent = first.parent
                parent_class = ' '.join(parent.get('class', [])) if parent else 'No parent'
                print(f"   First element parent: {parent_class[:50]}")
        
        # Save sample HTML for inspection
        print(f"\n💾 Saving sample HTML to debug_ebay.html")
        with open("debug_ebay.html", "w") as f:
            f.write(result.cleaned_html)

if __name__ == "__main__":
    asyncio.run(check_ebay_structure())