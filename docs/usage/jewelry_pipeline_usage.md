# Jewelry Extraction Pipeline - Usage Guide

## Overview

The Jewelry Extraction Pipeline is a comprehensive system for extracting jewelry listing data from eBay. It integrates multiple components to provide reliable, scalable, and intelligent data extraction with anti-bot detection, error handling, and image processing capabilities.

## Key Features

### Core Functionality
- **Single URL Extraction**: Extract data from individual eBay jewelry listings
- **Search Result Extraction**: Extract multiple listings from search results
- **Database Operations**: Save and retrieve listings with SQLite
- **Image Processing**: Download and process jewelry images concurrently
- **Data Quality Validation**: Comprehensive scoring and validation

### Advanced Features
- **Crawl4AI Integration**: Uses AsyncWebCrawler for reliable page rendering
- **Anti-Bot Detection**: Advanced user agent rotation, fingerprinting, and request patterns
- **Error Handling**: Circuit breakers, retry logic, and adaptive rate limiting
- **Selector Management**: Robust CSS selectors with fallbacks and success tracking
- **Performance Monitoring**: Real-time statistics and performance metrics

## Installation & Setup

### Prerequisites
```bash
# Install required dependencies
pip install crawl4ai aiohttp aiosqlite beautifulsoup4 pillow imagehash
pip install user-agents nltk pydantic
```

### Initialize NLTK Data (if using categorization)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Quick Start

### 1. Basic Single URL Extraction

```python
import asyncio
from jewelry_extraction_pipeline import extract_single_listing

async def extract_single():
    # Extract a single listing
    listing = await extract_single_listing(
        url="https://www.ebay.com/itm/123456789",
        config={
            'anti_detection': {
                'user_agents': {'rotation_frequency': 30},
                'request_patterns': {'min_delay': 2.0, 'max_delay': 5.0}
            }
        }
    )
    
    if listing:
        print(f"Title: {listing.title}")
        print(f"Price: ${listing.price}")
        print(f"Category: {listing.category.value}")
        print(f"Quality Score: {listing.data_quality_score:.2f}")

# Run the extraction
asyncio.run(extract_single())
```

### 2. Search Results Extraction

```python
import asyncio
from jewelry_extraction_pipeline import extract_search_results

async def extract_search():
    # Extract from search results
    listings = await extract_search_results(
        query="diamond engagement ring",
        max_pages=3,
        save_to_db=True
    )
    
    print(f"Found {len(listings)} listings")
    for listing in listings[:5]:  # Show first 5
        print(f"- {listing.title} - ${listing.price}")

asyncio.run(extract_search())
```

## Advanced Usage

### 1. Full Extractor Configuration

```python
import asyncio
from jewelry_extraction_pipeline import JewelryExtractor

async def advanced_extraction():
    # Comprehensive configuration
    config = {
        'anti_detection': {
            'user_agents': {
                'rotation_frequency': 25,
                'max_consecutive_failures': 3,
                'block_duration': 1800
            },
            'proxies': {
                'proxy_list': [
                    {
                        'address': 'proxy1.example.com',
                        'port': 8080,
                        'proxy_type': 'http',
                        'username': 'user',
                        'password': 'pass'
                    }
                ],
                'rotation_frequency': 20
            },
            'request_patterns': {
                'min_delay': 2.0,
                'max_delay': 8.0,
                'human_simulation': True,
                'burst_protection': True
            },
            'fingerprinting': {
                'randomize_viewport': True,
                'randomize_timezone': True,
                'randomize_language': True
            }
        },
        'error_handling': {
            'circuit_breaker_failure_threshold': 5,
            'circuit_breaker_recovery_timeout': 120,
            'default_rate_limit': 1.5,
            'max_error_records': 1000
        },
        'use_llm_extraction': False,  # Set to True for LLM-enhanced extraction
        'use_chunking': False
    }
    
    # Initialize extractor
    extractor = JewelryExtractor(
        config=config,
        database_path="./jewelry_database.db",
        images_directory="./jewelry_images",
        enable_anti_detection=True,
        enable_image_processing=True
    )
    
    await extractor.initialize()
    
    try:
        # Extract with progress tracking
        def progress_callback(progress, info):
            print(f"Progress: {progress:.1%} - Page {info['page']}, "
                  f"Listing {info['listing_index']}, Total: {info['total_listings']}")
        
        listings = await extractor.extract_from_search(
            query="vintage gold necklace",
            max_pages=5,
            category="necklaces",
            min_price=50.0,
            max_price=2000.0,
            progress_callback=progress_callback
        )
        
        # Save all listings
        for listing in listings:
            success = await extractor.save_to_database(listing)
            if success:
                print(f"Saved: {listing.title}")
        
        # Get detailed statistics
        stats = await extractor.get_statistics()
        print(f"\nExtraction Statistics:")
        print(f"- URLs processed: {stats['urls_processed']}")
        print(f"- Listings extracted: {stats['listings_extracted']}")
        print(f"- Images processed: {stats['images_processed']}")
        print(f"- Success rate: {stats['listings_extracted']/stats['urls_processed']:.2%}")
        
        # Print component statistics
        if 'image_processor_stats' in stats:
            img_stats = stats['image_processor_stats']
            print(f"- Concurrent features: {len(img_stats['concurrent_features'])}")
            print(f"- Processing stages: {img_stats['processing_stages_count']}")
        
    finally:
        await extractor.cleanup()

asyncio.run(advanced_extraction())
```

### 2. Custom Image Processing

```python
async def extract_with_custom_images():
    extractor = JewelryExtractor(
        enable_image_processing=True,
        images_directory="./custom_jewelry_images"
    )
    await extractor.initialize()
    
    try:
        # Extract listing
        listing = await extractor.extract_from_url("https://www.ebay.com/itm/123456")
        
        if listing and listing.image_urls:
            # Process images with custom settings
            success = await extractor.process_images(listing)
            
            if success:
                print(f"Processed {listing.image_count} images")
                # Access processed image metadata
                if 'processed_images' in listing.metadata:
                    for img in listing.metadata['processed_images']:
                        print(f"- {img['filename']}: Quality {img['quality_score']:.2f}")
        
    finally:
        await extractor.cleanup()
```

### 3. Database Operations

```python
async def database_operations():
    from jewelry_extraction_pipeline import DatabaseManager
    
    # Initialize database
    db_manager = DatabaseManager("./my_jewelry_db.db")
    await db_manager.initialize_database()
    
    # Retrieve specific listing
    listing = await db_manager.get_listing("listing_id_123")
    if listing:
        print(f"Found: {listing.title}")
    
    # Query database directly for complex searches
    import aiosqlite
    async with aiosqlite.connect("./my_jewelry_db.db") as db:
        cursor = await db.execute("""
            SELECT title, price, category, data_completeness_score 
            FROM jewelry_listings 
            WHERE category = 'rings' AND price BETWEEN 100 AND 1000
            ORDER BY data_completeness_score DESC
            LIMIT 10
        """)
        
        high_quality_rings = await cursor.fetchall()
        print("Top 10 high-quality rings:")
        for row in high_quality_rings:
            print(f"- {row[0]} - ${row[1]} (Quality: {row[3]:.2f})")
```

## Configuration Options

### Anti-Detection Configuration
```python
anti_detection_config = {
    'user_agents': {
        'rotation_frequency': 30,           # Rotate every N requests
        'max_consecutive_failures': 5,      # Block agent after N failures
        'block_duration': 3600              # Block duration in seconds
    },
    'proxies': {
        'proxy_list': [],                   # List of proxy configurations
        'rotation_frequency': 25,           # Rotate every N requests
        'health_check_interval': 300        # Health check interval
    },
    'request_patterns': {
        'min_delay': 1.0,                   # Minimum delay between requests
        'max_delay': 5.0,                   # Maximum delay between requests
        'human_simulation': True,           # Enable human-like interactions
        'burst_protection': True            # Prevent request bursts
    },
    'fingerprinting': {
        'randomize_viewport': True,         # Randomize browser viewport
        'randomize_timezone': True,         # Randomize timezone
        'randomize_language': True          # Randomize language settings
    }
}
```

### Error Handling Configuration
```python
error_handling_config = {
    'circuit_breaker_failure_threshold': 5,    # Failures before opening circuit
    'circuit_breaker_recovery_timeout': 60,    # Recovery timeout seconds
    'default_rate_limit': 2.0,                 # Default requests per second
    'max_error_records': 1000,                 # Maximum error records to keep
    'max_retries': 3                           # Maximum retry attempts
}
```

### Image Processing Configuration
```python
# Custom image processing (via ImageProcessor)
image_config = {
    'concurrency_mode': 'balanced',     # conservative, balanced, aggressive, adaptive
    'max_concurrent_downloads': 10,     # Override auto-detection
    'max_concurrent_processing': 4,     # Override auto-detection
    'request_delay': 0.5,              # Delay between image requests
    'timeout': 30,                     # Request timeout
    'max_retries': 3,                  # Retry attempts
    'memory_limit_mb': 512             # Memory limit
}
```

## Data Models

### JewelryListing Fields
```python
# Core required fields
id: str                    # Unique listing identifier
title: str                 # Listing title
price: float              # Current price
currency: str             # Currency code (USD, EUR, etc.)
condition: str            # Item condition
seller_id: str            # Seller username/ID
listing_url: str          # Full eBay listing URL
category: JewelryCategory # Jewelry category enum
material: JewelryMaterial # Primary material enum

# Optional enrichment fields
original_price: float     # Original/retail price
brand: str               # Brand name
gemstone: str            # Primary gemstone
size: str                # Size information
weight: str              # Weight information
description: str         # Full description
image_urls: List[str]    # List of image URLs
shipping_cost: float     # Shipping cost
seller_rating: float     # Seller feedback score

# Quality and metadata
data_quality_score: float    # Calculated quality score (0-1)
scraped_at: datetime         # When scraped
metadata: Dict[str, Any]     # Additional metadata
```

### Categories and Materials
```python
# Available categories
JewelryCategory: RINGS, NECKLACES, EARRINGS, BRACELETS, WATCHES, 
                BROOCHES, ANKLETS, PENDANTS, CHAINS, SETS, OTHER

# Available materials  
JewelryMaterial: GOLD, SILVER, PLATINUM, TITANIUM, STAINLESS_STEEL,
                COPPER, BRASS, LEATHER, FABRIC, PLASTIC, CERAMIC, 
                MIXED, UNKNOWN
```

## Performance Optimization

### 1. Concurrency Settings
```python
# For high-performance extraction
config = {
    'anti_detection': {
        'request_patterns': {
            'min_delay': 0.5,        # Faster requests
            'max_delay': 2.0,
            'burst_protection': False # Disable for speed
        }
    }
}

extractor = JewelryExtractor(
    config=config,
    enable_image_processing=True  # Concurrent image processing
)
```

### 2. Batch Processing
```python
async def batch_extract_urls(urls):
    extractor = JewelryExtractor()
    await extractor.initialize()
    
    # Process URLs in batches
    batch_size = 10
    results = []
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            extractor.extract_from_url(url) for url in batch
        ], return_exceptions=True)
        
        # Filter successful results
        valid_results = [r for r in batch_results if isinstance(r, JewelryListing)]
        results.extend(valid_results)
        
        # Delay between batches
        await asyncio.sleep(2)
    
    await extractor.cleanup()
    return results
```

## Error Handling and Monitoring

### 1. Error Statistics
```python
async def monitor_extraction():
    extractor = JewelryExtractor()
    await extractor.initialize()
    
    # Perform extractions...
    
    # Get comprehensive statistics
    stats = await extractor.get_statistics()
    
    # Check error rates
    error_rate = stats['errors_encountered'] / stats['urls_processed']
    if error_rate > 0.1:  # More than 10% errors
        print("High error rate detected!")
    
    # Check anti-detection stats
    if 'anti_detection_stats' in stats:
        detection_stats = stats['anti_detection_stats']
        if detection_stats['strategy_changes'] > 5:
            print("Multiple strategy adaptations - consider adjusting config")
    
    await extractor.cleanup()
```

### 2. Custom Error Handling
```python
from jewelry_extraction_pipeline import JewelryExtractor
from error_handling_system import with_error_handling

class CustomExtractor(JewelryExtractor):
    
    @with_error_handling("custom_extraction", use_circuit_breaker=True)
    async def extract_with_custom_logic(self, url: str):
        # Custom extraction logic with automatic error handling
        listing = await self.extract_from_url(url)
        
        # Custom validation
        if listing and listing.price < 10:
            raise ValueError("Price too low - likely extraction error")
        
        return listing
```

## Testing

### Run Tests
```bash
# Run the comprehensive test suite
python test_jewelry_pipeline.py
```

### Test Coverage
- Component initialization
- Database operations
- URL building
- Selector system
- Data extraction (mock)
- Image processing integration
- Error handling
- Data quality validation
- Statistics monitoring

## Best Practices

### 1. Rate Limiting
- Use appropriate delays between requests (2-5 seconds)
- Enable burst protection for large-scale scraping
- Monitor detection events and adapt accordingly

### 2. Data Quality
- Always check `data_quality_score` before using listings
- Validate required fields with `validate_for_database()`
- Use quality thresholds for filtering

### 3. Resource Management
- Always call `await extractor.cleanup()` when done
- Monitor memory usage for large batches
- Use database transactions for bulk operations

### 4. Compliance
- Respect robots.txt and site terms of service
- Implement appropriate delays and limits
- Monitor for detection and adapt behavior

## Troubleshooting

### Common Issues

1. **High Error Rates**
   - Increase delays between requests
   - Enable anti-detection features
   - Check selector patterns for updates

2. **Database Errors**
   - Ensure database is initialized
   - Check disk space and permissions
   - Validate data before saving

3. **Image Processing Failures**
   - Check image URL validity
   - Verify network connectivity
   - Monitor memory usage

4. **Detection Issues**
   - Enable user agent rotation
   - Use proxy rotation
   - Increase request delays

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug configuration
extractor = JewelryExtractor(
    config={'debug': True},
    enable_anti_detection=True
)
```

## Production Deployment

### 1. Configuration for Production
```python
production_config = {
    'anti_detection': {
        'user_agents': {'rotation_frequency': 20},
        'request_patterns': {
            'min_delay': 3.0,
            'max_delay': 8.0,
            'human_simulation': True
        }
    },
    'error_handling': {
        'circuit_breaker_failure_threshold': 3,
        'default_rate_limit': 0.5  # Conservative rate
    }
}
```

### 2. Monitoring and Logging
```python
import logging

# Production logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jewelry_extraction.log'),
        logging.StreamHandler()
    ]
)
```

### 3. Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_category_price ON jewelry_listings(category, price);
CREATE INDEX idx_quality_score ON jewelry_listings(data_completeness_score DESC);
CREATE INDEX idx_scraped_date ON jewelry_listings(scraped_at);
```

This comprehensive pipeline provides a robust foundation for jewelry data extraction with all the modern features needed for reliable, scalable scraping operations.