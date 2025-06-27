#!/usr/bin/env python3
"""
Test eBay selectors against real page to debug why 0 listings found
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

async def test_ebay_selectors():
    """Test selectors against real eBay jewelry search"""
    
    # Test URL - same as system generates
    test_url = "https://www.ebay.com/sch/i.html?_sacat=52546&_from=R40"  # rings category
    
    print(f"🔍 Testing URL: {test_url}")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(test_url)
        
        if not result.success:
            print(f"❌ Failed to fetch page: {result.error_message}")
            return
            
        print(f"✅ Page fetched successfully")
        print(f"📊 Content length: {len(result.cleaned_html)} characters")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(result.cleaned_html, 'html.parser')
        
        # Test current selectors
        selectors_to_test = [
            ('div.s-item', 'Current primary selector'),
            ('div[data-view*="mi:"]', 'Current fallback 1'),
            ('.srp-results .s-item', 'Current fallback 2'),
            ('div.s-item__wrapper', 'Current fallback 3'),
            ('li.s-item', 'Current fallback 4'),
            # Additional common eBay selectors
            ('.s-item', 'Generic s-item class'),
            ('[data-testid="item"]', 'Test ID selector'),
            ('article.s-item', 'Article s-item'),
            ('.srp-results li', 'SRP results list items'),
            ('.srp-results > div', 'SRP results divs')
        ]
        
        print("\n🧪 Testing selectors:")
        print("-" * 50)
        
        found_any = False
        for selector, description in selectors_to_test:
            elements = soup.select(selector)
            count = len(elements)
            status = "✅" if count > 0 else "❌"
            print(f"{status} {selector:<30} | {count:>3} found | {description}")
            
            if count > 0:
                found_any = True
                # Show first element HTML snippet
                first_element = elements[0]
                snippet = str(first_element)[:200] + "..." if len(str(first_element)) > 200 else str(first_element)
                print(f"    📝 First element: {snippet}")
                print()
        
        if not found_any:
            print("\n🔍 No listing selectors worked. Let's examine page structure...")
            
            # Look for common patterns
            common_patterns = [
                'div[class*="item"]',
                'div[class*="listing"]', 
                'div[class*="result"]',
                'li[class*="item"]',
                'article',
                '[data-testid]',
                '[data-view]'
            ]
            
            print("\n🔍 Checking common patterns:")
            for pattern in common_patterns:
                elements = soup.select(pattern)
                if elements:
                    print(f"✅ {pattern:<25} | {len(elements):>3} found")
                    
            # Show page title and some structure
            title = soup.find('title')
            print(f"\n📄 Page title: {title.text if title else 'Not found'}")
            
            # Look for any divs with classes containing 's-item'
            items_with_s_item = soup.find_all('div', class_=lambda x: x and 's-item' in x)
            print(f"🔍 Divs with 's-item' in class: {len(items_with_s_item)}")
            
            if items_with_s_item:
                first_item = items_with_s_item[0]
                print(f"📝 First s-item class: {first_item.get('class', [])}")

if __name__ == "__main__":
    asyncio.run(test_ebay_selectors())