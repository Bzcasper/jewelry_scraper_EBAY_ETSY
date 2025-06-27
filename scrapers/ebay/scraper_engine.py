"""
eBay Jewelry Scraper Engine Architecture

Comprehensive scraping engine with crawl4ai integration, anti-bot measures,
and modular design for reliable eBay jewelry data extraction.

Key Features:
- Anti-detection browser configuration
- Intelligent rate limiting and retry logic
- Modular component architecture
- Comprehensive error handling
- Data quality validation
- Concurrent processing support
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import uuid

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy, CosineStrategy
from crawl4ai.chunking_strategy import RegexChunking

from .ebay_selectors import SelectorManager, SelectorType, DeviceType, JewelryCategory

# Import jewelry models using relative imports
from ...models.jewelry_models import JewelryListing, JewelryImage, ScrapingSession, ScrapingStatus

from ...core.ebay_image_processor import ImageProcessor
from ...models.ebay_types import ScrapingMode, AntiDetectionLevel, ScrapingResult


@dataclass
class ScrapingConfig:
    """Comprehensive scraping configuration"""

    # Basic configuration
    max_concurrent_requests: int = 3
    request_delay_range: tuple = (2, 5)  # Random delay between requests
    max_retries: int = 3
    retry_delay_base: int = 5

    # Browser configuration
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080

    # Anti-detection settings
    anti_detection_level: AntiDetectionLevel = AntiDetectionLevel.STANDARD
    rotate_user_agents: bool = True
    simulate_human_behavior: bool = True
    use_proxy_rotation: bool = False
    proxy_list: List[str] = field(default_factory=list)

    # Data extraction settings
    extract_images: bool = True
    max_images_per_listing: int = 20
    image_quality_threshold: float = 0.7

    # Quality control
    min_data_quality_score: float = 0.5
    validate_data: bool = True
    skip_duplicates: bool = True

    # Output settings
    save_raw_html: bool = False
    save_screenshots: bool = False
    output_directory: str = "./output"

    # Session management
    session_timeout: int = 3600  # 1 hour
    max_listings_per_session: int = 1000

    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            key: value.value if isinstance(value, Enum) else value
            for key, value in self.__dict__.items()
        }


class EbayJewelryScraper:
    """
    Main eBay Jewelry Scraper Engine

    Orchestrates all scraping operations with intelligent anti-detection,
    quality control, and performance monitoring.
    """

    def __init__(self, config: Optional[ScrapingConfig] = None):
        """
        Initialize the scraper engine

        Args:
            config: Scraping configuration (uses defaults if None)
        """
        self.config = config or ScrapingConfig()
        self.session_id = str(uuid.uuid4())

        # Initialize components
        self.selector_manager = SelectorManager(
            enable_analytics=True,
            performance_monitoring=True
        )
        self.image_processor = ImageProcessor()

        # Initialize session tracking
        self.session = ScrapingSession(
            session_id=self.session_id,
            session_name=f"eBay Jewelry Scraping - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            rate_limit_delay=self.config.request_delay_range[0]
        )

        # Setup logging
        self.logger = self._setup_logger()

        # Initialize crawler
        self.crawler: Optional[AsyncWebCrawler] = None
        self._crawler_initialized = False

        # Performance tracking
        self.request_count = 0
        self.start_time = time.time()
        self.last_request_time = 0

        # User agent rotation
        self.user_agents = self._get_user_agents()
        self.current_user_agent_index = 0

        # Anti-detection state
        self.detection_risk_score = 0.0
        self.cooldown_until = None

        self.logger.info(
            f"Scraper initialized with session ID: {self.session_id}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(
            f"ebay_jewelry_scraper_{self.session_id[:8]}")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_user_agents(self) -> List[str]:
        """Get list of realistic user agents for rotation"""
        return [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0'
        ]

    async def _initialize_crawler(self):
        """Initialize the Crawl4AI crawler with anti-detection measures"""
        if self._crawler_initialized:
            return

        try:
            # Configure browser based on anti-detection level
            browser_config = self._create_browser_config()

            # Initialize crawler
            self.crawler = AsyncWebCrawler(
                verbose=True if self.config.log_level == "DEBUG" else False
            )

            await self.crawler.astart()
            self._crawler_initialized = True

            self.logger.info("Crawler initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize crawler: {e}")
            raise

    def _create_browser_config(self) -> BrowserConfig:
        """Create browser configuration with anti-detection measures"""

        # Base configuration
        config = BrowserConfig(
            browser_type=self.config.browser_type,
            headless=self.config.headless,
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height
        )

        # Anti-detection configurations based on level
        if self.config.anti_detection_level == AntiDetectionLevel.MINIMAL:
            config.extra_args = ['--no-sandbox', '--disable-dev-shm-usage']

        elif self.config.anti_detection_level == AntiDetectionLevel.STANDARD:
            config.extra_args = [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-features=TranslateUI'
            ]

        elif self.config.anti_detection_level == AntiDetectionLevel.AGGRESSIVE:
            config.extra_args = [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-features=TranslateUI',
                '--disable-plugins',
                '--disable-images',
                '--disable-javascript-harmony-shipping',
                '--disable-web-security',
                '--allow-running-insecure-content'
            ]

        elif self.config.anti_detection_level == AntiDetectionLevel.STEALTH:
            config.extra_args = [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-features=TranslateUI',
                '--disable-plugins',
                '--disable-web-security',
                '--allow-running-insecure-content',
                '--user-agent=' + self._get_next_user_agent()
            ]

        return config

    def _get_next_user_agent(self) -> str:
        """Get next user agent for rotation"""
        if self.config.rotate_user_agents:
            user_agent = self.user_agents[self.current_user_agent_index]
            self.current_user_agent_index = (
                self.current_user_agent_index + 1) % len(self.user_agents)
            return user_agent
        return self.user_agents[0]

    async def _apply_rate_limiting(self):
        """Apply intelligent rate limiting"""
        current_time = time.time()

        # Check if we're in cooldown period
        if self.cooldown_until and current_time < self.cooldown_until:
            cooldown_remaining = self.cooldown_until - current_time
            self.logger.warning(
                f"In cooldown period, waiting {cooldown_remaining:.1f} seconds")
            await asyncio.sleep(cooldown_remaining)

        # Calculate delay based on request history and risk score
        base_delay = random.uniform(*self.config.request_delay_range)

        # Increase delay if detection risk is high
        if self.detection_risk_score > 0.7:
            base_delay *= 2
            self.logger.warning(
                f"High detection risk ({self.detection_risk_score:.2f}), increasing delay")

        # Ensure minimum time between requests
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            if time_since_last < base_delay:
                additional_delay = base_delay - time_since_last
                await asyncio.sleep(additional_delay)

        self.last_request_time = time.time()

    async def _simulate_human_behavior(self):
        """Simulate human-like browsing behavior"""
        if not self.config.simulate_human_behavior:
            return

        # Random small delays to simulate reading/scrolling
        if random.random() < 0.3:  # 30% chance
            await asyncio.sleep(random.uniform(0.5, 2.0))

        # Occasionally longer pauses to simulate breaks
        if random.random() < 0.05:  # 5% chance
            await asyncio.sleep(random.uniform(10, 30))
            self.logger.debug("Simulated human break")

    async def scrape_search_results(self,
                                    search_url: str,
                                    max_pages: int = 10,
                                    max_listings: Optional[int] = None) -> ScrapingResult:
        """
        Scrape jewelry listings from eBay search results

        Args:
            search_url: eBay search URL
            max_pages: Maximum pages to scrape
            max_listings: Maximum listings to extract

        Returns:
            ScrapingResult with list of JewelryListing objects
        """
        start_time = time.time()

        try:
            await self._initialize_crawler()

            self.logger.info(f"Starting search results scraping: {search_url}")

            listings = []
            current_page = 1

            while current_page <= max_pages:
                if max_listings and len(listings) >= max_listings:
                    break

                # Build page URL
                page_url = self._build_page_url(search_url, current_page)

                # Apply rate limiting
                await self._apply_rate_limiting()

                # Scrape page
                page_result = await self._scrape_search_page(page_url)

                if not page_result.success:
                    self.logger.error(
                        f"Failed to scrape page {current_page}: {page_result.error}")
                    break

                page_listings = page_result.data or []
                listings.extend(page_listings)

                self.logger.info(
                    f"Page {current_page}: Found {len(page_listings)} listings")

                # Update session stats
                self.session.pages_processed += 1
                self.session.listings_found += len(page_listings)

                # Check if we should continue
                if len(page_listings) == 0:
                    self.logger.info("No more listings found, stopping")
                    break

                current_page += 1

                # Simulate human behavior
                await self._simulate_human_behavior()

            # Apply quality filtering
            if self.config.min_data_quality_score > 0:
                original_count = len(listings)
                listings = [
                    listing for listing in listings
                    if listing.data_quality_score >= self.config.min_data_quality_score
                ]
                filtered_count = original_count - len(listings)
                if filtered_count > 0:
                    self.logger.info(
                        f"Filtered out {filtered_count} low-quality listings")

            # Update session stats
            self.session.listings_scraped = len(listings)
            self.session.status = ScrapingStatus.COMPLETED

            execution_time = time.time() - start_time

            return ScrapingResult(
                success=True,
                data=listings,
                response_time=execution_time,
                quality_score=self._calculate_batch_quality_score(listings),
                metadata={
                    'pages_scraped': current_page - 1,
                    'total_listings': len(listings),
                    'session_id': self.session_id
                }
            )

        except Exception as e:
            self.logger.error(f"Search results scraping failed: {e}")
            self.session.status = ScrapingStatus.FAILED
            self.session.last_error = str(e)

            return ScrapingResult(
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )

    def _build_page_url(self, base_url: str, page: int) -> str:
        """Build URL for specific page number"""
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}_pgn={page}"

    async def _scrape_search_page(self, url: str) -> ScrapingResult:
        """Scrape a single search results page"""
        start_time = time.time()

        try:
            # Configure crawl parameters
            run_config = CrawlerRunConfig(
                cache_mode="bypass",
                wait_for_images=False,
                screenshot=self.config.save_screenshots,
                process_iframes=False,
                remove_overlay_elements=True,
                simulate_user=True,
                override_navigator=True
            )

            # Execute crawl
            result = await self.crawler.arun(url, config=run_config)

            if not result.success:
                return ScrapingResult(
                    success=False,
                    error=f"Crawl failed: {result.error_message}",
                    response_time=time.time() - start_time
                )

            # Extract listing data
            listings = await self._extract_listings_from_page(result.cleaned_html, url)

            # Update risk score based on response
            self._update_detection_risk(result)

            return ScrapingResult(
                success=True,
                data=listings,
                response_time=time.time() - start_time,
                metadata={
                    'status_code': result.status_code,
                    'response_size': len(result.html),
                    'listings_extracted': len(listings)
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to scrape page {url}: {e}")
            return ScrapingResult(
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )

    async def _extract_listings_from_page(self, html: str, page_url: str) -> List[JewelryListing]:
        """Extract jewelry listings from search results page HTML"""
        listings = []

        try:
            # Use BeautifulSoup for parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Get listing containers
            container_selector = self.selector_manager.get_selector(
                SelectorType.SEARCH_RESULTS,
                'listing_container'
            )

            if not container_selector:
                self.logger.error("No listing container selector found")
                return listings

            # Find all listing containers
            containers = soup.select(container_selector.primary)

            if not containers:
                # Try fallback selectors
                for fallback in container_selector.fallbacks:
                    containers = soup.select(fallback)
                    if containers:
                        break

            self.logger.info(f"Found {len(containers)} listing containers")

            for container in containers:
                try:
                    listing = await self._extract_listing_from_container(container, page_url)
                    if listing:
                        listings.append(listing)
                except Exception as e:
                    self.logger.warning(f"Failed to extract listing: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Failed to parse page HTML: {e}")

        return listings

    async def _extract_listing_from_container(self, container, page_url: str) -> Optional[JewelryListing]:
        """Extract jewelry listing data from a container element"""
        try:
            # Generate unique ID
            listing_id = str(uuid.uuid4())

            # Extract title
            title_elem = self._find_element_by_selectors(
                container,
                self.selector_manager.get_all_selectors('listing_title')
            )
            title = title_elem.get_text(strip=True) if title_elem else ""

            if not title:
                return None

            # Extract price
            price_elem = self._find_element_by_selectors(
                container,
                self.selector_manager.get_all_selectors('listing_price')
            )
            price = self._extract_price(
                price_elem.get_text(strip=True) if price_elem else "")

            # Extract URL
            url_elem = self._find_element_by_selectors(
                container,
                self.selector_manager.get_all_selectors('listing_url')
            )
            listing_url = url_elem.get('href', '') if url_elem else ""

            # Make URL absolute
            if listing_url and not listing_url.startswith('http'):
                listing_url = f"https://www.ebay.com{listing_url}"

            # Extract condition
            condition_elem = self._find_element_by_selectors(
                container,
                self.selector_manager.get_all_selectors('listing_condition')
            )
            condition = condition_elem.get_text(
                strip=True) if condition_elem else "Unknown"

            # Extract image URL
            image_elem = self._find_element_by_selectors(
                container,
                self.selector_manager.get_all_selectors('listing_image')
            )
            image_url = image_elem.get('src', '') if image_elem else ""

            # Basic jewelry categorization
            category = self._categorize_jewelry(title)
            material = self._extract_material(title)

            # Create listing object
            listing = JewelryListing(
                id=listing_id,
                title=title,
                price=price,
                condition=condition,
                seller_id="extracted_from_search",  # Will be filled by individual listing scraper
                listing_url=listing_url,
                category=category,
                material=material,
                image_urls=[image_url] if image_url else [],
                scraped_at=datetime.now(),
                metadata={
                    'source_page': page_url,
                    'extraction_method': 'search_results'
                }
            )

            # Calculate and update quality score
            listing.update_quality_score()

            return listing

        except Exception as e:
            self.logger.error(f"Failed to extract listing from container: {e}")
            return None

    def _find_element_by_selectors(self, container, selectors: List[str]):
        """Find element using multiple selectors as fallbacks"""
        for selector in selectors:
            try:
                element = container.select_one(selector)
                if element:
                    return element
            except Exception:
                continue
        return None

    def _extract_price(self, price_text: str) -> float:
        """Extract numeric price from text"""
        import re

        if not price_text:
            return 0.0

        # Remove currency symbols and extract numbers
        price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
        if price_match:
            try:
                return float(price_match.group())
            except ValueError:
                pass

        return 0.0

    def _categorize_jewelry(self, title: str) -> JewelryCategory:
        """Basic jewelry categorization based on title"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['ring', 'wedding', 'engagement', 'band']):
            return JewelryCategory.RINGS
        elif any(word in title_lower for word in ['necklace', 'chain', 'pendant']):
            return JewelryCategory.NECKLACES
        elif any(word in title_lower for word in ['earring', 'stud', 'hoop']):
            return JewelryCategory.EARRINGS
        elif any(word in title_lower for word in ['bracelet', 'bangle']):
            return JewelryCategory.BRACELETS
        elif any(word in title_lower for word in ['watch', 'timepiece']):
            return JewelryCategory.WATCHES
        elif any(word in title_lower for word in ['brooch', 'pin']):
            return JewelryCategory.BROOCHES
        else:
            return JewelryCategory.OTHER

    def _extract_material(self, title: str):
        """Extract material from title"""
        from ...models.jewelry_models import JewelryMaterial

        title_lower = title.lower()

        if any(word in title_lower for word in ['gold', '14k', '18k', '10k', '24k']):
            return JewelryMaterial.GOLD
        elif any(word in title_lower for word in ['silver', 'sterling', '.925']):
            return JewelryMaterial.SILVER
        elif 'platinum' in title_lower:
            return JewelryMaterial.PLATINUM
        elif 'titanium' in title_lower:
            return JewelryMaterial.TITANIUM
        elif any(word in title_lower for word in ['steel', 'stainless']):
            return JewelryMaterial.STAINLESS_STEEL
        else:
            return JewelryMaterial.UNKNOWN

    def _update_detection_risk(self, crawl_result):
        """Update detection risk score based on response"""
        # Factors that increase detection risk:
        # - Unusual response times
        # - Missing expected elements
        # - Blocked/redirected responses
        # - Rate limiting responses

        risk_factors = 0

        # Check response time (very fast responses might indicate blocking)
        if hasattr(crawl_result, 'response_time') and crawl_result.response_time < 0.5:
            risk_factors += 0.2

        # Check for blocking indicators in HTML
        if hasattr(crawl_result, 'html'):
            html_lower = crawl_result.html.lower()
            if any(indicator in html_lower for indicator in [
                'blocked', 'captcha', 'robot', 'automation', 'suspicious'
            ]):
                risk_factors += 0.4

        # Check status code
        if hasattr(crawl_result, 'status_code'):
            if crawl_result.status_code == 429:  # Rate limited
                risk_factors += 0.6
            elif crawl_result.status_code >= 400:
                risk_factors += 0.3

        # Update risk score with exponential decay
        self.detection_risk_score = (
            self.detection_risk_score * 0.8) + (risk_factors * 0.2)

        # Trigger cooldown if risk is very high
        if self.detection_risk_score > 0.8:
            cooldown_time = 60 * \
                (1 + self.detection_risk_score)  # 60-120 seconds
            self.cooldown_until = time.time() + cooldown_time
            self.logger.warning(
                f"High detection risk, entering cooldown for {cooldown_time:.0f}s")

    def _calculate_batch_quality_score(self, listings: List[JewelryListing]) -> float:
        """Calculate overall quality score for a batch of listings"""
        if not listings:
            return 0.0

        total_score = sum(listing.data_quality_score for listing in listings)
        return total_score / len(listings)

    async def scrape_individual_listing(self, listing_url: str) -> ScrapingResult:
        """
        Scrape detailed information from an individual jewelry listing

        Args:
            listing_url: URL of the eBay listing

        Returns:
            ScrapingResult with detailed JewelryListing object
        """
        # This will be implemented in scraper_007
        # For now, return a placeholder
        return ScrapingResult(
            success=False,
            error="Individual listing scraping not yet implemented"
        )

    async def close(self):
        """Clean up resources"""
        if self.crawler and self._crawler_initialized:
            await self.crawler.aclose()
            self._crawler_initialized = False

        # Save selector performance cache
        self.selector_manager.save_cache()

        self.logger.info("Scraper closed successfully")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            'session_id': self.session_id,
            'duration': self.session.duration.total_seconds() if self.session.duration else 0,
            'listings_found': self.session.listings_found,
            'listings_scraped': self.session.listings_scraped,
            'pages_processed': self.session.pages_processed,
            'requests_made': self.request_count,
            'detection_risk_score': self.detection_risk_score,
            'success_rate': self.session.success_rate,
            'status': self.session.status.value
        }

    def __repr__(self) -> str:
        return f"EbayJewelryScraper(session_id={self.session_id[:8]}...)"
