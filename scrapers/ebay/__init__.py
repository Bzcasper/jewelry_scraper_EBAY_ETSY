"""
eBay Jewelry Scraper Module

Comprehensive eBay jewelry scraping system with advanced anti-detection,
intelligent rate limiting, robust error handling, and high-quality data extraction.

Main Components:
- EbayJewelryScraper: Main scraping engine with crawl4ai integration
- JewelryListingScraper: Individual listing data extraction
- SelectorManager: CSS selector management with fallbacks
- AdvancedBrowserConfigurator: Anti-detection browser configuration
- EbayJewelryURLBuilder: Dynamic search URL construction
- AdvancedRateLimiter: Intelligent rate limiting and retry logic
- AdvancedErrorHandler: Comprehensive error handling and recovery
- ImageProcessor: Image downloading and processing

Usage Example:
    from crawl4ai.crawlers.ebay_jewelry import EbayJewelryScraper, ScrapingConfig
    
    config = ScrapingConfig(
        anti_detection_level="standard",
        max_concurrent_requests=2,
        extract_images=True
    )
    
    scraper = EbayJewelryScraper(config)
    
    # Search for jewelry
    search_url = "https://www.ebay.com/sch/i.html?_nkw=diamond+ring"
    results = await scraper.scrape_search_results(search_url, max_pages=5)
    
    # Individual listing
    listing_result = await scraper.scrape_individual_listing(listing_url)
    
    await scraper.close()
"""

from .scraper_engine import (
    EbayJewelryScraper,
    ScrapingConfig
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.ebay_types import (
    ScrapingResult,
    ScrapingMode,
    AntiDetectionLevel
)

from .listing_scraper import (
    JewelryListingScraper,
    ExtractionContext
)

from .ebay_selectors import (
    SelectorManager,
    SelectorType,
    SelectorInfo,
    DeviceType,
    JewelryCategory
)

from core.browser_config import (
    AdvancedBrowserConfigurator,
    BrowserFingerprint,
    DeviceProfile,
    ProxyConfig,
    ViewportConfig
)

from ..ebay_url_builder import (
    EBayURLBuilder,
    SearchFilters,
    SortOrder,
    ListingType,
    ItemCondition
)

from utils.rate_limiter import (
    AdvancedRateLimiter,
    RateLimitConfig,
    RetryConfig,
    RetryPolicy,
    RateLimitStrategy,
    create_ebay_rate_limiter
)

from utils.ebay_error_handler import (
    AdvancedErrorHandler,
    ErrorClassification,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy
)

from core.ebay_image_processor import (
    ImageProcessor,
    ImageMetadata
)

# Version information
__version__ = "1.0.0"
__author__ = "eBay Jewelry Scraper Team"

# Export all main classes
__all__ = [
    # Main scraper
    'EbayJewelryScraper',
    'ScrapingConfig',
    'ScrapingResult',
    'ScrapingMode',
    'AntiDetectionLevel',

    # Listing scraper
    'JewelryListingScraper',
    'ExtractionContext',

    # Selectors
    'SelectorManager',
    'SelectorType',
    'SelectorInfo',
    'DeviceType',
    'JewelryCategory',

    # Browser configuration
    'AdvancedBrowserConfigurator',
    'BrowserFingerprint',
    'DeviceProfile',
    'ProxyConfig',
    'ViewportConfig',

    # URL building
    'EBayURLBuilder',
    'SearchFilters',
    'SortOrder',
    'ListingType',
    'ItemCondition',

    # Rate limiting
    'AdvancedRateLimiter',
    'RateLimitConfig',
    'RetryConfig',
    'RetryPolicy',
    'RateLimitStrategy',
    'create_ebay_rate_limiter',

    # Error handling
    'AdvancedErrorHandler',
    'ErrorClassification',
    'ErrorContext',
    'ErrorCategory',
    'ErrorSeverity',
    'RecoveryStrategy',

    # Image processing
    'ImageProcessor',
    'ImageMetadata'
]


def create_standard_scraper(**kwargs) -> EbayJewelryScraper:
    """
    Create a standard eBay jewelry scraper with recommended settings

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        Configured EbayJewelryScraper instance
    """
    config = ScrapingConfig(
        anti_detection_level=AntiDetectionLevel.STANDARD,
        max_concurrent_requests=2,
        request_delay_range=(2, 5),
        extract_images=True,
        validate_data=True,
        **kwargs
    )

    return EbayJewelryScraper(config)


def create_stealth_scraper(**kwargs) -> EbayJewelryScraper:
    """
    Create a stealth eBay jewelry scraper with maximum anti-detection

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        Configured EbayJewelryScraper instance
    """
    config = ScrapingConfig(
        anti_detection_level=AntiDetectionLevel.STEALTH,
        max_concurrent_requests=1,
        request_delay_range=(5, 10),
        simulate_human_behavior=True,
        rotate_user_agents=True,
        extract_images=True,
        validate_data=True,
        **kwargs
    )

    return EbayJewelryScraper(config)


def create_fast_scraper(**kwargs) -> EbayJewelryScraper:
    """
    Create a fast eBay jewelry scraper with minimal delays (higher detection risk)

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        Configured EbayJewelryScraper instance
    """
    config = ScrapingConfig(
        anti_detection_level=AntiDetectionLevel.MINIMAL,
        max_concurrent_requests=5,
        request_delay_range=(0.5, 2),
        simulate_human_behavior=False,
        extract_images=False,  # Skip images for speed
        validate_data=False,   # Skip validation for speed
        **kwargs
    )

    return EbayJewelryScraper(config)
