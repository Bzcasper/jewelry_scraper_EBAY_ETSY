#!/usr/bin/env python3
"""
Test different eBay URL formats to find what works
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from crawl4ai import AsyncWebCrawler

async def test_ebay_urls():
    """Test different eBay URL formats"""
    
    # Different URL formats to test
    test_urls = [
        ("Current system URL", "https://www.ebay.com/sch/i.html?_sacat=52546&_from=R40"),
        ("Simple search URL", "https://www.ebay.com/sch/i.html?_nkw=ring+jewelry"),
        ("Category + keyword", "https://www.ebay.com/sch/i.html?_sacat=52546&_nkw=ring"),
        ("Working example", "https://www.ebay.com/sch/i.html?_nkw=diamond+ring&_sacat=0"),
        ("Direct category", "https://www.ebay.com/b/Rings/52546/bn_1853252")
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for description, url in test_urls:
            print(f"\n🔍 Testing: {description}")
            print(f"URL: {url}")
            
            result = await crawler.arun(url)
            
            if result.success:
                # Check if we got a search results page
                if "Search results" in result.cleaned_html or "results for" in result.cleaned_html.lower():
                    print("✅ Valid search results page")
                    
                    # Quick test for listings
                    if "s-item" in result.cleaned_html:
                        print("✅ Contains s-item elements")
                    else:
                        print("❌ No s-item elements found")
                        
                elif "eBay Home" in result.cleaned_html:
                    print("❌ Redirected to eBay Home")
                else:
                    print("❓ Unknown page type")
                    
                # Check content length
                print(f"📊 Content length: {len(result.cleaned_html)} chars")
                
                # Look for title
                title_start = result.cleaned_html.find("<title>")
                title_end = result.cleaned_html.find("</title>")
                if title_start >= 0 and title_end >= 0:
                    title = result.cleaned_html[title_start+7:title_end]
                    print(f"📄 Title: {title[:100]}...")
            else:
                print(f"❌ Failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(test_ebay_urls())