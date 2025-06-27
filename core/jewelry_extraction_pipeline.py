"""
Comprehensive Jewelry Extraction Pipeline for eBay Listings
Integrates all components for complete jewelry data extraction with Crawl4AI

This module provides the main JewelryExtractor class that coordinates:
- Crawl4AI AsyncWebCrawler integration
- Advanced selector management with fallbacks
- Image processing and download
- Database operations with validation
- Error handling and retry logic
- Anti-bot detection measures
- Rate limiting and request management
"""

import asyncio
import logging
import json
import hashlib
import sqlite3
import aiofiles
import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import re
import random

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# Local component imports
from ..models.jewelry_models import (
    JewelryListing, JewelryCategory, JewelryMaterial, ListingStatus,
    JewelryImage, ImageType, ScrapingSession, ScrapingStatus,
    JEWELRY_SCHEMA_SQL, JEWELRY_INDEXES_SQL
)
from .image_pipeline import ImageProcessor, ConcurrencyMode
from ..scrapers.ebay.ebay_selectors import SelectorManager, SelectorType, DeviceType
from ..utils.anti_detection_system import AntiDetectionSystem, RequestType
from ..utils.error_handling_system import ErrorManager, with_error_handling


class DatabaseManager:
    """Database operations manager for jewelry listings"""

    def __init__(self, db_path: str = "./jewelry_database.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    async def initialize_database(self):
        """Initialize database schema and indexes"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create tables
                for table_name, schema in JEWELRY_SCHEMA_SQL.items():
                    await db.execute(schema)

                # Create indexes
                for index_sql in JEWELRY_INDEXES_SQL:
                    await db.execute(index_sql)

                await db.commit()
                self.logger.info("Database initialized successfully")

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    async def save_listing(self, listing: JewelryListing) -> bool:
        """Save jewelry listing to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Convert listing to database format
                insert_sql = """
                    INSERT OR REPLACE INTO jewelry_listings (
                        listing_id, url, title, price, original_price, currency,
                        condition, availability, seller_name, seller_rating,
                        seller_feedback_count, category, subcategory, brand,
                        material, materials, size, weight, dimensions,
                        main_stone, stone_color, stone_clarity, stone_cut,
                        stone_carat, accent_stones, description, features,
                        tags, item_number, listing_type, listing_status,
                        watchers, views, bids, time_left, shipping_cost,
                        ships_from, ships_to, image_count, description_length,
                        data_completeness_score, created_at, updated_at,
                        scraped_at, listing_date, metadata, raw_data,
                        is_validated, validation_errors
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                values = (
                    listing.id, listing.listing_url, listing.title, listing.price,
                    getattr(listing, 'original_price', None), listing.currency,
                    listing.condition, getattr(listing, 'availability', None),
                    listing.seller_id, getattr(listing, 'seller_rating', None),
                    getattr(listing, 'seller_feedback_count', None),
                    listing.category.value, getattr(
                        listing, 'subcategory', None),
                    listing.brand, listing.material.value,
                    json.dumps(getattr(listing, 'materials', [])),
                    listing.size, listing.weight, getattr(
                        listing, 'dimensions', None),
                    listing.gemstone, getattr(listing, 'stone_color', None),
                    getattr(listing, 'stone_clarity', None), getattr(
                        listing, 'stone_cut', None),
                    getattr(listing, 'stone_carat', None),
                    json.dumps(getattr(listing, 'accent_stones', [])),
                    getattr(listing, 'description', None),
                    json.dumps(getattr(listing, 'features', [])),
                    json.dumps(getattr(listing, 'tags', [])),
                    getattr(listing, 'item_number', None),
                    getattr(listing, 'listing_type', None),
                    getattr(listing, 'listing_status',
                            ListingStatus.UNKNOWN).value,
                    getattr(listing, 'watchers', None), getattr(
                        listing, 'views', None),
                    getattr(listing, 'bids', None), getattr(
                        listing, 'time_left', None),
                    listing.shipping_cost, getattr(
                        listing, 'ships_from', None),
                    getattr(listing, 'ships_to', None), len(
                        listing.image_urls),
                    len(getattr(listing, 'description', '') or ''),
                    listing.data_quality_score, listing.created_at, listing.updated_at,
                    listing.scraped_at, getattr(listing, 'listing_date', None),
                    json.dumps(getattr(listing, 'metadata', {})),
                    json.dumps(getattr(listing, 'raw_data', {})),
                    getattr(listing, 'is_validated', False),
                    json.dumps(getattr(listing, 'validation_errors', []))
                )

                await db.execute(insert_sql, values)
                await db.commit()

                self.logger.info(f"Saved listing: {listing.id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save listing {listing.id}: {e}")
            return False

    async def get_listing(self, listing_id: str) -> Optional[JewelryListing]:
        """Retrieve listing by ID"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM jewelry_listings WHERE listing_id = ?",
                    (listing_id,)
                )
                row = await cursor.fetchone()

                if row:
                    # Convert database row back to JewelryListing
                    return self._row_to_listing(dict(row))

                return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve listing {listing_id}: {e}")
            return None

    def _row_to_listing(self, row: Dict[str, Any]) -> JewelryListing:
        """Convert database row to JewelryListing object"""
        # Parse JSON fields
        materials = json.loads(row.get('materials', '[]'))
        accent_stones = json.loads(row.get('accent_stones', '[]'))
        features = json.loads(row.get('features', '[]'))
        tags = json.loads(row.get('tags', '[]'))
        metadata = json.loads(row.get('metadata', '{}'))
        raw_data = json.loads(row.get('raw_data', '{}'))
        validation_errors = json.loads(row.get('validation_errors', '[]'))

        # Create JewelryListing instance
        return JewelryListing(
            id=row['listing_id'],
            title=row['title'],
            price=row['price'],
            currency=row['currency'],
            condition=row['condition'],
            seller_id=row['seller_name'],
            listing_url=row['url'],
            end_time=datetime.fromisoformat(
                row['end_time']) if row.get('end_time') else None,
            shipping_cost=row.get('shipping_cost'),
            category=JewelryCategory(row['category']),
            material=JewelryMaterial(row['material']),
            gemstone=row['main_stone'],
            size=row['size'],
            weight=row['weight'],
            brand=row['brand'],
            image_urls=row.get('image_urls', '').split(
                ',') if row.get('image_urls') else [],
            main_image_path=row.get('main_image_path'),
            scraped_at=datetime.fromisoformat(
                row['scraped_at']) if row['scraped_at'] else datetime.now(),
            data_quality_score=row.get('data_completeness_score', 0.0),
            listing_id=row.get('item_number'),
            original_price=row.get('original_price'),
            availability=row.get('availability'),
            seller_rating=row.get('seller_rating'),
            seller_feedback_count=row.get('seller_feedback_count'),
            subcategory=row.get('subcategory'),
            dimensions=row.get('dimensions'),
            stone_color=row.get('stone_color'),
            stone_clarity=row.get('stone_clarity'),
            stone_cut=row.get('stone_cut'),
            stone_carat=row.get('stone_carat'),
            description=row.get('description'),
            item_number=row.get('item_number'),
            listing_type=row.get('listing_type'),
            watchers=row.get('watchers'),
            views=row.get('views'),
            bids=row.get('bids'),
            time_left=row.get('time_left'),
            ships_from=row.get('ships_from'),
            ships_to=row.get('ships_to'),
            listing_date=datetime.fromisoformat(
                row['listing_date']) if row.get('listing_date') else None,
            materials=materials,
            accent_stones=accent_stones,
            features=features,
            tags=tags,
            metadata=metadata,
            raw_data=raw_data,
            validation_errors=validation_errors
        )


class URLBuilder:
    """eBay URL builder for search queries"""

    @staticmethod
    def build_search_url(query: str,
                         category: Optional[str] = None,
                         min_price: Optional[float] = None,
                         max_price: Optional[float] = None,
                         condition: Optional[str] = None,
                         page: int = 1,
                         sort: str = "BestMatch") -> str:
        """Build eBay search URL with parameters"""

        base_url = "https://www.ebay.com/sch/i.html"
        params = {
            "_nkw": query,
            "_pgn": page,
            "_sop": {"BestMatch": "12", "PriceAsc": "15", "PriceDesc": "16", "EndingSoon": "1"}.get(sort, "12")
        }

        # Add category filter
        if category:
            # Jewelry category codes
            category_codes = {
                "rings": "10968",
                "necklaces": "4182",
                "earrings": "10984",
                "bracelets": "10977",
                "watches": "14324"
            }
            if category.lower() in category_codes:
                params["_sacat"] = category_codes[category.lower()]

        # Add price filters
        if min_price:
            params["_udlo"] = str(min_price)
        if max_price:
            params["_udhi"] = str(max_price)

        # Add condition filter
        if condition:
            condition_codes = {
                "new": "1000",
                "used": "3000",
                "refurbished": "2000"
            }
            if condition.lower() in condition_codes:
                params["LH_ItemCondition"] = condition_codes[condition.lower()]

        # Build URL
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{param_string}"


class JewelryExtractor:
    """
    Main jewelry extraction coordinator integrating all components

    Provides high-level interface for:
    - Single URL extraction
    - Search result extraction  
    - Database operations
    - Image processing
    - Error handling and recovery
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 database_path: str = "./jewelry_database.db",
                 images_directory: str = "./images",
                 enable_anti_detection: bool = True,
                 enable_image_processing: bool = True):
        """
        Initialize the jewelry extractor

        Args:
            config: Configuration dictionary for various components
            database_path: SQLite database file path
            images_directory: Directory for image storage
            enable_anti_detection: Enable anti-bot detection measures
            enable_image_processing: Enable image download and processing
        """

        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.db_manager = DatabaseManager(database_path)
        self.selector_manager = SelectorManager()
        self.url_builder = URLBuilder()
        self.error_manager = ErrorManager(
            self.config.get('error_handling', {}))

        # Initialize image processor if enabled
        self.enable_image_processing = enable_image_processing
        if enable_image_processing:
            self.image_processor = ImageProcessor(
                base_directory=images_directory,
                concurrency_mode=ConcurrencyMode.BALANCED,
                enable_performance_monitoring=True
            )

        # Initialize anti-detection system if enabled
        self.enable_anti_detection = enable_anti_detection
        if enable_anti_detection:
            self.anti_detection = AntiDetectionSystem(
                self.config.get('anti_detection', {}))

        # Crawl4AI configuration
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security"
            ]
        )

        # Statistics tracking
        self.stats = {
            'urls_processed': 0,
            'listings_extracted': 0,
            'listings_saved': 0,
            'images_processed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'end_time': None
        }

    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize database
            await self.db_manager.initialize_database()

            # Initialize image processor
            if self.enable_image_processing:
                # Image processor is already initialized
                pass

            self.logger.info("JewelryExtractor initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize JewelryExtractor: {e}")
            raise

    @with_error_handling("extract_from_url")
    async def extract_from_url(self, url: str) -> Optional[JewelryListing]:
        """
        Extract jewelry listing data from a single eBay URL

        Args:
            url: eBay listing URL

        Returns:
            JewelryListing object or None if extraction failed
        """

        self.stats['urls_processed'] += 1

        try:
            # Prepare request with anti-detection
            request_config = None
            if self.enable_anti_detection:
                request_config = await self.anti_detection.prepare_request(
                    RequestType.LISTING_PAGE, mobile=False
                )

            # Configure Crawl4AI run
            run_config = CrawlerRunConfig(
                word_count_threshold=100,
                extraction_strategy=LLMExtractionStrategy(
                    provider="ollama/llama3.2",
                    api_token="",
                    instruction="Extract jewelry listing details including title, price, condition, seller info, and specifications"
                ) if self.config.get('use_llm_extraction') else None,
                chunking_strategy=RegexChunking() if self.config.get('use_chunking') else None,
                bypass_cache=True,
                process_iframes=False,
                remove_overlay_elements=True,
                simulate_user=True,
                override_navigator=True,
                magic=True
            )

            # Apply anti-detection user agent if available
            if request_config and 'user_agent' in request_config:
                run_config.user_agent = request_config['user_agent']

            # Execute crawl with Crawl4AI
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=run_config
                )

                if not result.success:
                    self.logger.error(
                        f"Failed to crawl {url}: {result.error_message}")
                    return None

                # Extract listing data from HTML
                listing = await self._extract_listing_from_html(
                    html_content=result.html,
                    url=url,
                    metadata={
                        'crawl_time': result.response_headers.get('date'),
                        'status_code': result.status_code,
                        'response_headers': dict(result.response_headers) if result.response_headers else {}
                    }
                )

                if listing:
                    self.stats['listings_extracted'] += 1

                    # Process images if enabled
                    if self.enable_image_processing and listing.image_urls:
                        await self.process_images(listing)

                    return listing

                return None

        except Exception as e:
            self.stats['errors_encountered'] += 1
            self.logger.error(f"Error extracting from URL {url}: {e}")
            return None

    @with_error_handling("extract_from_search")
    async def extract_from_search(self,
                                  query: str,
                                  max_pages: int = 5,
                                  category: Optional[str] = None,
                                  min_price: Optional[float] = None,
                                  max_price: Optional[float] = None,
                                  progress_callback: Optional[Callable] = None) -> List[JewelryListing]:
        """
        Extract jewelry listings from eBay search results

        Args:
            query: Search query string
            max_pages: Maximum pages to scrape
            category: Jewelry category filter
            min_price: Minimum price filter
            max_price: Maximum price filter
            progress_callback: Optional progress callback function

        Returns:
            List of JewelryListing objects
        """

        self.stats['start_time'] = datetime.now()
        listings = []

        try:
            for page in range(1, max_pages + 1):
                self.logger.info(f"Processing search page {page}/{max_pages}")

                # Build search URL
                search_url = self.url_builder.build_search_url(
                    query=query,
                    category=category,
                    min_price=min_price,
                    max_price=max_price,
                    page=page
                )

                # Get listing URLs from search page
                listing_urls = await self._extract_listing_urls_from_search(search_url)

                if not listing_urls:
                    self.logger.warning(f"No listings found on page {page}")
                    break

                # Extract data from each listing
                for i, listing_url in enumerate(listing_urls):
                    listing = await self.extract_from_url(listing_url)
                    if listing:
                        listings.append(listing)

                    # Progress callback
                    if progress_callback:
                        total_processed = (page - 1) * \
                            len(listing_urls) + i + 1
                        estimated_total = max_pages * len(listing_urls)
                        progress = total_processed / estimated_total
                        progress_callback(progress, {
                            'page': page,
                            'listing_index': i,
                            'total_listings': len(listings),
                            'current_url': listing_url
                        })

                    # Rate limiting between requests
                    if self.enable_anti_detection:
                        delay = random.uniform(1.0, 3.0)
                        await asyncio.sleep(delay)

                # Rate limiting between pages
                if page < max_pages:
                    delay = random.uniform(3.0, 7.0)
                    await asyncio.sleep(delay)

            self.stats['end_time'] = datetime.now()
            self.logger.info(
                f"Search extraction completed. Found {len(listings)} listings")

            return listings

        except Exception as e:
            self.stats['errors_encountered'] += 1
            self.logger.error(f"Error in search extraction: {e}")
            return listings

    async def save_to_database(self, listing: JewelryListing) -> bool:
        """
        Save jewelry listing to database

        Args:
            listing: JewelryListing object to save

        Returns:
            True if saved successfully, False otherwise
        """

        try:
            # Validate listing data
            if not listing.validate_for_database():
                self.logger.warning(
                    f"Listing validation failed: {listing.validation_errors}")
                return False

            # Update quality score
            listing.update_quality_score()

            # Save to database
            success = await self.db_manager.save_listing(listing)

            if success:
                self.stats['listings_saved'] += 1
                self.logger.info(f"Saved listing to database: {listing.id}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving listing to database: {e}")
            return False

    async def process_images(self, listing: JewelryListing) -> bool:
        """
        Process and download images for a listing

        Args:
            listing: JewelryListing object with image URLs

        Returns:
            True if processing successful, False otherwise
        """

        if not self.enable_image_processing or not listing.image_urls:
            return False

        try:
            # Process images with the image processor
            processed_metadata = await self.image_processor.process_urls_batch(
                urls=listing.image_urls[:10],  # Limit to first 10 images
                batch_size=5
            )

            # Update listing with image processing results
            listing.image_count = len(processed_metadata)
            self.stats['images_processed'] += len(processed_metadata)

            # Update metadata
            if not hasattr(listing, 'metadata') or listing.metadata is None:
                listing.metadata = {}

            listing.metadata['processed_images'] = [
                {
                    'filename': meta.filename,
                    'quality_score': meta.quality_metrics.overall_score if meta.quality_metrics else 0.0,
                    'file_size': meta.file_size,
                    'dimensions': meta.dimensions
                }
                for meta in processed_metadata
            ]

            self.logger.info(
                f"Processed {len(processed_metadata)} images for listing {listing.id}")
            return True

        except Exception as e:
            self.logger.error(
                f"Error processing images for listing {listing.id}: {e}")
            return False

    async def _extract_listing_urls_from_search(self, search_url: str) -> List[str]:
        """Extract listing URLs from search results page"""

        try:
            # Prepare request with anti-detection
            request_config = None
            if self.enable_anti_detection:
                request_config = await self.anti_detection.prepare_request(
                    RequestType.SEARCH_PAGE, mobile=False
                )

            run_config = CrawlerRunConfig(
                word_count_threshold=50,
                bypass_cache=True,
                process_iframes=False,
                remove_overlay_elements=True,
                simulate_user=True
            )

            if request_config and 'user_agent' in request_config:
                run_config.user_agent = request_config['user_agent']

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=search_url, config=run_config)

                if not result.success:
                    self.logger.error(
                        f"Failed to crawl search page {search_url}")
                    return []

                # Extract listing URLs using selectors
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(result.html, 'html.parser')

                urls = []
                selectors = self.selector_manager.get_all_selectors(
                    'listing_url', DeviceType.DESKTOP, include_fallbacks=True
                )

                for selector in selectors:
                    try:
                        elements = soup.select(selector)
                        for element in elements:
                            href = element.get('href')
                            if href and 'itm' in href:
                                # Clean and validate URL
                                if href.startswith('http'):
                                    urls.append(href)
                                elif href.startswith('/'):
                                    urls.append(f"https://www.ebay.com{href}")

                        if urls:
                            break  # Found URLs with this selector

                    except Exception as e:
                        self.logger.debug(f"Selector {selector} failed: {e}")
                        continue

                # Remove duplicates and clean URLs
                unique_urls = []
                seen = set()
                for url in urls:
                    # Extract item ID to deduplicate
                    item_id_match = re.search(r'/itm/([^/?]+)', url)
                    if item_id_match:
                        item_id = item_id_match.group(1)
                        if item_id not in seen:
                            seen.add(item_id)
                            unique_urls.append(url)

                self.logger.info(
                    f"Found {len(unique_urls)} unique listings on search page")
                return unique_urls[:50]  # Limit to 50 per page

        except Exception as e:
            self.logger.error(
                f"Error extracting listing URLs from search: {e}")
            return []

    async def _extract_listing_from_html(self,
                                         html_content: str,
                                         url: str,
                                         metadata: Dict[str, Any]) -> Optional[JewelryListing]:
        """Extract listing data from HTML content"""

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Generate listing ID
            listing_id = self._generate_listing_id(url)

            # Initialize listing with required fields
            listing = JewelryListing(
                id=listing_id,
                title="",
                price=0.0,
                currency="USD",
                condition="Unknown",
                seller_id="unknown",
                listing_url=url,
                end_time=None,
                shipping_cost=None,
                category=JewelryCategory.OTHER,
                material=JewelryMaterial.UNKNOWN,
                gemstone=None,
                size=None,
                weight=None,
                brand=None,
                main_image_path=None,
                scraped_at=datetime.now(),
                data_quality_score=0.0,
                listing_id=None,
                original_price=None,
                availability=None,
                seller_rating=None,
                seller_feedback_count=None,
                subcategory=None,
                dimensions=None,
                stone_color=None,
                stone_clarity=None,
                stone_cut=None,
                stone_carat=None,
                description=None,
                item_number=None,
                listing_type=None,
                watchers=None,
                views=None,
                bids=None,
                time_left=None,
                ships_from=None,
                ships_to=None,
                listing_date=None
            )

            # Extract title
            title_selectors = self.selector_manager.get_all_selectors(
                'title', DeviceType.DESKTOP)
            for selector in title_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        listing.title = element.get_text(strip=True)
                        break
                except:
                    continue

            # Extract price
            price_selectors = self.selector_manager.get_all_selectors(
                'current_price', DeviceType.DESKTOP)
            for selector in price_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        price_text = element.get_text(strip=True)
                        price_match = re.search(
                            r'[\d,]+\.?\d*', price_text.replace(',', ''))
                        if price_match:
                            listing.price = float(price_match.group())
                            break
                except:
                    continue

            # Extract condition
            condition_selectors = self.selector_manager.get_all_selectors(
                'condition', DeviceType.DESKTOP)
            for selector in condition_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        listing.condition = element.get_text(strip=True)
                        break
                except:
                    continue

            # Extract seller information
            seller_selectors = self.selector_manager.get_all_selectors(
                'seller_name', DeviceType.DESKTOP)
            for selector in seller_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        listing.seller_id = element.get_text(strip=True)
                        break
                except:
                    continue

            # Extract images
            image_selectors = self.selector_manager.get_all_selectors(
                'gallery_thumbnails', DeviceType.DESKTOP)
            image_urls = []
            for selector in image_selectors:
                try:
                    elements = soup.select(selector)
                    for img in elements:
                        src = img.get('src') or img.get('data-src')
                        if src and src.startswith('http'):
                            # Get high resolution version
                            high_res_src = src.replace(
                                's-l64', 's-l1600').replace('s-l140', 's-l1600')
                            image_urls.append(high_res_src)

                    if image_urls:
                        break
                except:
                    continue

            listing.image_urls = list(set(image_urls))  # Remove duplicates

            # Basic categorization based on title
            listing.category = self._categorize_from_title(listing.title)
            listing.material = self._extract_material_from_title(listing.title)

            # Set metadata
            listing.metadata = metadata
            listing.metadata['extraction_timestamp'] = datetime.now(
            ).isoformat()
            listing.metadata['html_length'] = len(html_content)

            # Update quality score
            listing.update_quality_score()

            # Validate required fields
            if not listing.title or listing.price <= 0:
                self.logger.warning(
                    f"Listing missing required data: {listing_id}")
                return None

            return listing

        except Exception as e:
            self.logger.error(f"Error extracting listing data from HTML: {e}")
            return None

    def _generate_listing_id(self, url: str) -> str:
        """Generate unique listing ID from URL"""
        # Try to extract eBay item ID
        item_id_match = re.search(r'/itm/([^/?]+)', url)
        if item_id_match:
            return item_id_match.group(1)

        # Fallback to URL hash
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _categorize_from_title(self, title: str) -> JewelryCategory:
        """Basic categorization from title keywords"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['ring', 'band', 'engagement', 'wedding']):
            return JewelryCategory.RINGS
        elif any(word in title_lower for word in ['necklace', 'chain', 'pendant', 'choker']):
            return JewelryCategory.NECKLACES
        elif any(word in title_lower for word in ['earring', 'stud', 'hoop', 'drop']):
            return JewelryCategory.EARRINGS
        elif any(word in title_lower for word in ['bracelet', 'bangle', 'cuff']):
            return JewelryCategory.BRACELETS
        elif any(word in title_lower for word in ['watch', 'timepiece']):
            return JewelryCategory.WATCHES
        else:
            return JewelryCategory.OTHER

    def _extract_material_from_title(self, title: str) -> JewelryMaterial:
        """Basic material extraction from title"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['gold', '14k', '18k', '10k']):
            return JewelryMaterial.GOLD
        elif any(word in title_lower for word in ['silver', 'sterling', '925']):
            return JewelryMaterial.SILVER
        elif 'platinum' in title_lower:
            return JewelryMaterial.PLATINUM
        elif 'titanium' in title_lower:
            return JewelryMaterial.TITANIUM
        elif any(word in title_lower for word in ['steel', 'stainless']):
            return JewelryMaterial.STAINLESS_STEEL
        else:
            return JewelryMaterial.UNKNOWN

    async def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = self.stats.copy()

        # Calculate runtime
        if stats['start_time'] and stats['end_time']:
            runtime = (stats['end_time'] - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['listings_per_minute'] = stats['listings_extracted'] / \
                (runtime / 60) if runtime > 0 else 0

        # Add component statistics
        if self.enable_image_processing:
            stats['image_processor_stats'] = self.image_processor.get_statistics()

        if self.enable_anti_detection:
            stats['anti_detection_stats'] = self.anti_detection.get_statistics()

        stats['error_manager_stats'] = self.error_manager.get_error_statistics()

        return stats

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.enable_image_processing:
                # Save caches and cleanup
                await self.image_processor.__aexit__(None, None, None)

            self.logger.info("JewelryExtractor cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Convenience functions
async def extract_single_listing(url: str,
                                 config: Optional[Dict[str, Any]] = None) -> Optional[JewelryListing]:
    """
    Extract a single jewelry listing from eBay URL

    Args:
        url: eBay listing URL
        config: Optional configuration

    Returns:
        JewelryListing object or None
    """

    extractor = JewelryExtractor(config=config)
    await extractor.initialize()

    try:
        listing = await extractor.extract_from_url(url)
        return listing
    finally:
        await extractor.cleanup()


async def extract_search_results(query: str,
                                 max_pages: int = 5,
                                 config: Optional[Dict[str, Any]] = None,
                                 save_to_db: bool = True) -> List[JewelryListing]:
    """
    Extract jewelry listings from eBay search results

    Args:
        query: Search query
        max_pages: Maximum pages to scrape
        config: Optional configuration
        save_to_db: Whether to save results to database

    Returns:
        List of JewelryListing objects
    """

    extractor = JewelryExtractor(config=config)
    await extractor.initialize()

    try:
        listings = await extractor.extract_from_search(query, max_pages)

        if save_to_db:
            for listing in listings:
                await extractor.save_to_database(listing)

        return listings
    finally:
        await extractor.cleanup()


# Test and demo functions
async def demo_extraction():
    """Demo function showing extractor usage"""

    # Configuration
    config = {
        'anti_detection': {
            'user_agents': {'rotation_frequency': 30},
            'request_patterns': {'min_delay': 2.0, 'max_delay': 5.0}
        },
        'error_handling': {
            'circuit_breaker_failure_threshold': 3,
            'default_rate_limit': 1.0
        }
    }

    # Initialize extractor
    extractor = JewelryExtractor(
        config=config,
        enable_anti_detection=True,
        enable_image_processing=True
    )

    await extractor.initialize()

    try:
        # Example 1: Extract single listing
        print("Extracting single listing...")
        test_url = "https://www.ebay.com/itm/14K-Gold-Diamond-Ring-Engagement-Wedding-Band/155123456789"
        listing = await extractor.extract_from_url(test_url)

        if listing:
            print(f"Extracted: {listing.title} - ${listing.price}")
            await extractor.save_to_database(listing)

        # Example 2: Search extraction
        print("\nExtracting search results...")
        listings = await extractor.extract_from_search(
            query="diamond ring",
            max_pages=2,
            category="rings",
            min_price=100.0,
            max_price=1000.0
        )

        print(f"Found {len(listings)} listings")

        # Save all listings
        for listing in listings:
            await extractor.save_to_database(listing)

        # Show statistics
        stats = await extractor.get_statistics()
        print(f"\nExtraction Statistics:")
        print(f"- URLs processed: {stats['urls_processed']}")
        print(f"- Listings extracted: {stats['listings_extracted']}")
        print(f"- Listings saved: {stats['listings_saved']}")
        print(f"- Images processed: {stats['images_processed']}")
        print(f"- Errors encountered: {stats['errors_encountered']}")

    finally:
        await extractor.cleanup()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demo
    asyncio.run(demo_extraction())
