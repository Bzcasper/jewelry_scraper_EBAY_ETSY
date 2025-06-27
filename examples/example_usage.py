"""
eBay Jewelry Scraper - Complete Usage Example

This example demonstrates how to use the comprehensive eBay jewelry scraping system
with all its advanced features including anti-detection, rate limiting, error handling,
and data validation.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from crawl4ai.crawlers.ebay_jewelry import (
    create_standard_scraper,
    create_stealth_scraper,
    EbayJewelryURLBuilder,
    JewelrySearchFilters,
    PriceRange,
    SortOrder,
    JewelryCategory,
    ScrapingConfig,
    AntiDetectionLevel
)


async def main():
    """Main example function demonstrating the scraper capabilities"""
    
    print("🔍 eBay Jewelry Scraper - Complete Example")
    print("=" * 50)
    
    # Example 1: Basic Search Scraping
    await example_basic_search()
    
    # Example 2: Advanced Search with Filters
    await example_advanced_search()
    
    # Example 3: Individual Listing Scraping
    await example_individual_listing()
    
    # Example 4: Stealth Scraping with Maximum Anti-Detection
    await example_stealth_scraping()
    
    # Example 5: Custom Configuration
    await example_custom_configuration()
    
    print("\n✅ All examples completed successfully!")


async def example_basic_search():
    """Example 1: Basic search results scraping"""
    
    print("\n📋 Example 1: Basic Search Results Scraping")
    print("-" * 40)
    
    # Create a standard scraper
    scraper = create_standard_scraper()
    
    try:
        # Simple search for diamond rings
        search_url = "https://www.ebay.com/sch/i.html?_nkw=diamond+ring"
        
        print(f"🔗 Searching: {search_url}")
        
        # Scrape search results (limited to 2 pages for example)
        results = await scraper.scrape_search_results(
            search_url=search_url,
            max_pages=2,
            max_listings=20
        )
        
        if results.success:
            listings = results.data
            print(f"✅ Successfully scraped {len(listings)} listings")
            print(f"⚡ Quality score: {results.quality_score:.2f}")
            print(f"⏱️  Response time: {results.response_time:.2f}s")
            
            # Show first few listings
            for i, listing in enumerate(listings[:3]):
                print(f"\n📱 Listing {i+1}:")
                print(f"   Title: {listing.title[:80]}...")
                print(f"   Price: {listing.currency} {listing.price}")
                print(f"   Category: {listing.category.value}")
                print(f"   Material: {listing.material.value}")
                print(f"   Quality: {listing.data_quality_score:.1%}")
        else:
            print(f"❌ Scraping failed: {results.error}")
        
        # Show session statistics
        stats = scraper.get_session_stats()
        print(f"\n📊 Session Stats:")
        print(f"   Requests made: {stats['requests_made']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Detection risk: {stats['detection_risk_score']:.2f}")
        
    finally:
        await scraper.close()


async def example_advanced_search():
    """Example 2: Advanced search with comprehensive filters"""
    
    print("\n🔧 Example 2: Advanced Search with Filters")
    print("-" * 40)
    
    # Create URL builder for advanced search
    url_builder = EbayJewelryURLBuilder()
    
    # Define search filters
    filters = JewelrySearchFilters(
        keywords="gold necklace",
        price_range=PriceRange(min_price=50.0, max_price=500.0),
        material="gold",
        condition=ItemCondition.NEW,
        sort_order=SortOrder.PRICE_LOW_TO_HIGH,
        results_per_page=50,
        buy_it_now_only=True
    )
    
    # Build search URL
    search_url = url_builder.build_search_url(filters)
    print(f"🔗 Advanced search URL: {search_url}")
    
    # Create scraper with custom settings
    scraper = create_standard_scraper(
        extract_images=True,
        min_data_quality_score=0.6
    )
    
    try:
        results = await scraper.scrape_search_results(
            search_url=search_url,
            max_pages=3
        )
        
        if results.success:
            listings = results.data
            print(f"✅ Found {len(listings)} high-quality listings")
            
            # Analyze results
            categories = {}
            materials = {}
            price_range = {"min": float('inf'), "max": 0}
            
            for listing in listings:
                # Count categories
                cat = listing.category.value
                categories[cat] = categories.get(cat, 0) + 1
                
                # Count materials
                mat = listing.material.value
                materials[mat] = materials.get(mat, 0) + 1
                
                # Track price range
                if listing.price > 0:
                    price_range["min"] = min(price_range["min"], listing.price)
                    price_range["max"] = max(price_range["max"], listing.price)
            
            print(f"\n📈 Analysis:")
            print(f"   Categories: {categories}")
            print(f"   Materials: {materials}")
            print(f"   Price range: ${price_range['min']:.2f} - ${price_range['max']:.2f}")
            
    finally:
        await scraper.close()


async def example_individual_listing():
    """Example 3: Individual listing detailed extraction"""
    
    print("\n🎯 Example 3: Individual Listing Extraction")
    print("-" * 40)
    
    # Real eBay jewelry listing URL
    listing_url = "https://www.ebay.com/itm/14K-White-Gold-Diamond-Engagement-Ring-1-2-CT-TW-Size-7-Round-Solitaire/155234567890"
    
    scraper = create_standard_scraper()
    
    try:
        print(f"🔗 Extracting: {listing_url}")
        
        # Scrape individual listing
        result = await scraper.scrape_individual_listing(listing_url)
        
        if result.success:
            listing = result.data
            print(f"✅ Successfully extracted listing")
            
            # Show detailed information
            summary = listing.get_summary()
            print(f"\n📋 Listing Summary:")
            for key, value in summary.items():
                print(f"   {key.title()}: {value}")
            
            # Show specifications if available
            if hasattr(listing, 'raw_data') and 'specifications' in listing.raw_data:
                specs = listing.raw_data['specifications']
                if specs:
                    print(f"\n🔧 Specifications ({len(specs)}):")
                    for spec in specs[:5]:  # Show first 5
                        print(f"   {spec['attribute_name']}: {spec['attribute_value']}")
            
            # Show image information
            if listing.image_urls:
                print(f"\n🖼️  Images: {len(listing.image_urls)} URLs found")
                print(f"   Main image: {listing.main_image_path}")
        
        else:
            print(f"❌ Extraction failed: {result.error}")
            
    finally:
        await scraper.close()


async def example_stealth_scraping():
    """Example 4: Stealth scraping with maximum anti-detection"""
    
    print("\n🥷 Example 4: Stealth Scraping Mode")
    print("-" * 40)
    
    # Create stealth scraper
    scraper = create_stealth_scraper(
        simulate_human_behavior=True,
        rotate_user_agents=True
    )
    
    try:
        # Build conservative search
        url_builder = EbayJewelryURLBuilder()
        filters = JewelrySearchFilters(
            keywords="vintage jewelry",
            results_per_page=25  # Smaller page size for stealth
        )
        search_url = url_builder.build_search_url(filters)
        
        print(f"🔗 Stealth search: {search_url}")
        print("⏳ Using maximum stealth settings (slower but safer)...")
        
        results = await scraper.scrape_search_results(
            search_url=search_url,
            max_pages=2  # Limited for stealth
        )
        
        if results.success:
            print(f"✅ Stealth scraping successful: {len(results.data)} listings")
            
            # Show anti-detection effectiveness
            stats = scraper.get_session_stats()
            print(f"🛡️  Detection risk score: {stats['detection_risk_score']:.3f}")
            print(f"⏱️  Average delay per request: {results.response_time / stats['requests_made']:.1f}s")
        
    finally:
        await scraper.close()


async def example_custom_configuration():
    """Example 5: Custom scraper configuration"""
    
    print("\n⚙️  Example 5: Custom Configuration")
    print("-" * 40)
    
    # Create custom configuration
    config = ScrapingConfig(
        # Anti-detection settings
        anti_detection_level=AntiDetectionLevel.AGGRESSIVE,
        simulate_human_behavior=True,
        rotate_user_agents=True,
        
        # Performance settings
        max_concurrent_requests=1,  # Very conservative
        request_delay_range=(3, 8),  # Longer delays
        
        # Quality settings
        extract_images=True,
        max_images_per_listing=10,
        min_data_quality_score=0.7,
        validate_data=True,
        
        # Monitoring
        enable_monitoring=True,
        log_level="DEBUG"
    )
    
    # Create scraper with custom config
    scraper = EbayJewelryScraper(config)
    
    try:
        # Test with a targeted search
        url_builder = EbayJewelryURLBuilder()
        search_url = url_builder.build_category_url(
            "fine_rings",
            JewelrySearchFilters(
                price_range=PriceRange(min_price=100.0, max_price=1000.0),
                sort_order=SortOrder.PRICE_HIGH_TO_LOW
            )
        )
        
        print(f"🔗 Custom config search: {search_url}")
        
        results = await scraper.scrape_search_results(
            search_url=search_url,
            max_pages=1,
            max_listings=10
        )
        
        if results.success:
            high_quality_listings = [
                l for l in results.data 
                if l.data_quality_score >= 0.7
            ]
            
            print(f"✅ Found {len(high_quality_listings)} high-quality listings")
            print(f"📊 Average quality: {sum(l.data_quality_score for l in results.data) / len(results.data):.1%}")
            
            # Show the best listing
            if high_quality_listings:
                best_listing = max(high_quality_listings, key=lambda x: x.data_quality_score)
                print(f"\n🏆 Best Quality Listing ({best_listing.data_quality_score:.1%}):")
                print(f"   Title: {best_listing.title}")
                print(f"   Price: {best_listing.currency} {best_listing.price}")
                print(f"   Brand: {best_listing.brand or 'Unknown'}")
                print(f"   Images: {len(best_listing.image_urls)}")
        
    finally:
        await scraper.close()


async def save_results_example():
    """Example: Save results to files"""
    
    print("\n💾 Saving Results Example")
    print("-" * 40)
    
    scraper = create_standard_scraper()
    
    try:
        # Simple search
        results = await scraper.scrape_search_results(
            "https://www.ebay.com/sch/i.html?_nkw=jewelry",
            max_pages=1,
            max_listings=5
        )
        
        if results.success:
            # Create output directory
            output_dir = Path("ebay_jewelry_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            json_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to JSON-serializable format
            json_data = {
                'scraping_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_listings': len(results.data),
                    'quality_score': results.quality_score,
                    'response_time': results.response_time
                },
                'listings': [listing.to_dict() for listing in results.data]
            }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {json_file}")
            print(f"📊 Saved {len(results.data)} listings")
            
    finally:
        await scraper.close()


if __name__ == "__main__":
    # Run the complete example
    asyncio.run(main())
    
    # Uncomment to run save results example
    # asyncio.run(save_results_example())