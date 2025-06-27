"""
eBay Jewelry Scraper System

A comprehensive, production-ready solution for extracting jewelry listing data
from eBay. Built on top of Crawl4AI framework with anti-bot measures, automated
categorization, image processing, and MCP integration.

Main modules:
- core: Core processing components (extraction pipeline, image processing)
- scrapers: Platform-specific scrapers (eBay)
- models: Data models and schemas
- data: Database management and analytics
- utils: Utility functions and helpers
- cli: Command-line interfaces
- mcp: Model Context Protocol integration
- tests: Test suites
"""

from .core.jewelry_extraction_pipeline import JewelryExtractor
from .models.jewelry_models import JewelryListing, JewelryCategory, JewelryMaterial
from .data.jewelry_data_manager import JewelryDatabaseManager

__version__ = "1.0.0"
__author__ = "Jewelry Scraper Team"

__all__ = [
    'JewelryExtractor',
    'JewelryListing',
    'JewelryCategory',
    'JewelryMaterial',
    'JewelryDatabaseManager'
]
