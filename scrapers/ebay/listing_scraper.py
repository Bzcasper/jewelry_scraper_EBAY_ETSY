"""
Individual eBay Jewelry Listing Scraper

Specialized scraper for extracting detailed information from individual eBay jewelry listings.
Handles complex product pages with comprehensive data extraction and validation.

Features:
- Detailed product information extraction
- Image URL collection and validation
- Seller information processing
- Specification parsing
- Quality scoring and validation
- Error handling and retry logic
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag
import uuid

from crawl4ai import CrawlerRunConfig
from .ebay_selectors import SelectorManager, SelectorType, DeviceType, JewelryCategory

# Import from root-level jewelry models - use absolute import
import sys
from pathlib import Path
# Import jewelry models using relative imports
from ...models.jewelry_models import JewelryListing, JewelryImage, JewelrySpecification, JewelryMaterial, ListingStatus

from ...models.ebay_types import ScrapingResult


@dataclass
class ExtractionContext:
    """Context for data extraction operations"""
    url: str
    html: str
    soup: BeautifulSoup
    listing_id: str
    extraction_time: datetime
    quality_threshold: float = 0.5


class JewelryListingScraper:
    """
    Individual jewelry listing scraper with comprehensive data extraction
    """

    def __init__(self, selector_manager: SelectorManager):
        """
        Initialize listing scraper

        Args:
            selector_manager: CSS selector manager instance
        """
        self.selector_manager = selector_manager
        self.logger = logging.getLogger(__name__)

        # Price extraction patterns
        self.price_patterns = [
            r'\$[\d,]+\.?\d*',  # $123.45, $1,234
            r'USD\s*[\d,]+\.?\d*',  # USD 123.45
            r'[\d,]+\.?\d*\s*USD',  # 123.45 USD
            r'[\d,]+\.?\d*'  # 123.45 (fallback)
        ]

        # Material detection patterns
        self.material_patterns = {
            JewelryMaterial.GOLD: [
                r'\b(?:gold|14k|18k|10k|24k|yellow\s+gold|white\s+gold|rose\s+gold)\b',
                r'\bgold\s+filled\b',
                r'\bgold\s+plated\b'
            ],
            JewelryMaterial.SILVER: [
                r'\b(?:silver|sterling|\.925|sterling\s+silver)\b',
                r'\bsilver\s+plated\b'
            ],
            JewelryMaterial.PLATINUM: [
                r'\b(?:platinum|plat|pt950|pt900)\b'
            ],
            JewelryMaterial.TITANIUM: [
                r'\b(?:titanium|titanium\s+alloy)\b'
            ],
            JewelryMaterial.STAINLESS_STEEL: [
                r'\b(?:stainless\s+steel|steel|surgical\s+steel)\b'
            ]
        }

        # Gemstone patterns
        self.gemstone_patterns = [
            r'\b(?:diamond|diamonds|brilliant|solitaire)\b',
            r'\b(?:ruby|rubies|sapphire|emerald|topaz|amethyst|garnet|pearl|pearls)\b',
            r'\b(?:cubic\s+zirconia|cz|moissanite|opal|turquoise|jade)\b'
        ]

        # Size extraction patterns
        self.size_patterns = {
            'ring': [
                r'\bsize\s*[:=]?\s*(\d+(?:\.\d+)?)\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:size|sz)\b',
                r'\bUS\s*(\d+(?:\.\d+)?)\b'
            ],
            'chain': [
                r'\b(\d+(?:\.\d+)?)\s*(?:inch|inches|in|")\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b'
            ],
            'bracelet': [
                r'\b(\d+(?:\.\d+)?)\s*(?:inch|inches|in|")\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b'
            ]
        }

    async def scrape_listing(self,
                             url: str,
                             html: str,
                             quality_threshold: float = 0.5) -> ScrapingResult:
        """
        Scrape individual jewelry listing

        Args:
            url: Listing URL
            html: Page HTML content
            quality_threshold: Minimum quality threshold

        Returns:
            ScrapingResult with JewelryListing data
        """
        start_time = datetime.now()

        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')

            # Create extraction context
            context = ExtractionContext(
                url=url,
                html=html,
                soup=soup,
                listing_id=self._extract_listing_id(url),
                extraction_time=start_time,
                quality_threshold=quality_threshold
            )

            # Extract all data components
            listing_data = await self._extract_comprehensive_data(context)

            # Validate and create listing
            listing = self._create_jewelry_listing(listing_data, context)

            # Calculate quality score
            listing.update_quality_score()

            # Validate quality threshold
            if listing.data_quality_score < quality_threshold:
                self.logger.warning(
                    f"Listing quality score ({listing.data_quality_score:.2f}) "
                    f"below threshold ({quality_threshold})"
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            return ScrapingResult(
                success=True,
                data=listing,
                response_time=execution_time,
                quality_score=listing.data_quality_score,
                metadata={
                    'extraction_fields': len([k for k, v in listing_data.items() if v]),
                    'image_count': len(listing_data.get('image_urls', [])),
                    'specifications_count': len(listing_data.get('specifications', []))
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to scrape listing {url}: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return ScrapingResult(
                success=False,
                error=str(e),
                response_time=execution_time
            )

    def _extract_listing_id(self, url: str) -> str:
        """Extract eBay listing ID from URL"""
        # Try to extract from URL patterns
        patterns = [
            r'/itm/.*?/(\d+)',  # /itm/item-name/123456789
            r'/itm/(\d+)',      # /itm/123456789
            r'item=(\d+)',      # ?item=123456789
            r'#(\d+)'           # #123456789
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Fallback to UUID if no ID found
        return str(uuid.uuid4())

    async def _extract_comprehensive_data(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract all available data from listing page"""

        data = {}

        # Basic listing information
        data.update(await self._extract_basic_info(context))

        # Price information
        data.update(await self._extract_price_info(context))

        # Product details
        data.update(await self._extract_product_details(context))

        # Jewelry-specific attributes
        data.update(await self._extract_jewelry_attributes(context))

        # Images
        data.update(await self._extract_images(context))

        # Seller information
        data.update(await self._extract_seller_info(context))

        # Shipping information
        data.update(await self._extract_shipping_info(context))

        # Specifications and item specifics
        data.update(await self._extract_specifications(context))

        # Additional metadata
        data.update(await self._extract_metadata(context))

        return data

    async def _extract_basic_info(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract basic listing information"""

        data = {}

        # Title
        title_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('title')
        )
        data['title'] = self._clean_text(
            title_elem.get_text() if title_elem else "")

        # Condition
        condition_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('condition')
        )
        data['condition'] = self._clean_text(
            condition_elem.get_text() if condition_elem else "Unknown")

        # Item number
        item_number = self._extract_item_number(context.soup)
        data['item_number'] = item_number

        # Listing type
        data['listing_type'] = self._determine_listing_type(context.soup)

        # Status
        data['listing_status'] = self._determine_listing_status(context.soup)

        # Availability
        data['availability'] = self._extract_availability(context.soup)

        return data

    async def _extract_price_info(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract price information"""

        data = {}

        # Current price
        price_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('current_price')
        )

        if price_elem:
            price_text = self._clean_text(price_elem.get_text())
            data['price'] = self._extract_price_value(price_text)
            data['currency'] = self._extract_currency(price_text)
        else:
            data['price'] = 0.0
            data['currency'] = "USD"

        # Original price (if on sale)
        original_price_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('original_price')
        )

        if original_price_elem:
            original_price_text = self._clean_text(
                original_price_elem.get_text())
            data['original_price'] = self._extract_price_value(
                original_price_text)

        return data

    async def _extract_product_details(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract detailed product information"""

        data = {}

        # Description
        description = self._extract_description(context.soup)
        data['description'] = description
        data['description_length'] = len(description) if description else 0

        # Features (extract from description and bullet points)
        data['features'] = self._extract_features(context.soup, description)

        # Dimensions
        data['dimensions'] = self._extract_dimensions(description or "")

        # Weight
        data['weight'] = self._extract_weight(description or "")

        return data

    async def _extract_jewelry_attributes(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract jewelry-specific attributes"""

        data = {}
        title = context.soup.find('h1')
        title_text = title.get_text() if title else ""
        description = data.get('description', '')
        combined_text = f"{title_text} {description}".lower()

        # Material
        data['material'] = self._detect_primary_material(combined_text)
        data['materials'] = self._detect_all_materials(combined_text)

        # Gemstone
        data['gemstone'] = self._detect_primary_gemstone(combined_text)

        # Category
        data['category'] = self._categorize_jewelry(title_text)

        # Size (ring size, chain length, etc.)
        data['size'] = self._extract_size(combined_text, data['category'])

        # Brand
        brand_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('brand')
        )
        data['brand'] = self._clean_text(
            brand_elem.get_text() if brand_elem else "")

        # Stone-specific attributes
        stone_attrs = self._extract_stone_attributes(combined_text)
        data.update(stone_attrs)

        return data

    async def _extract_images(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract image URLs"""

        image_urls = []

        # Main image
        main_img_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('main_image')
        )

        if main_img_elem:
            main_url = main_img_elem.get(
                'src') or main_img_elem.get('data-src')
            if main_url:
                image_urls.append(self._normalize_image_url(main_url))

        # Gallery thumbnails
        thumbnail_elems = context.soup.select(
            self.selector_manager.get_selector(
                SelectorType.IMAGES, 'gallery_thumbnails').primary
        )

        for thumb in thumbnail_elems:
            thumb_url = thumb.get('src') or thumb.get('data-src')
            if thumb_url:
                normalized_url = self._normalize_image_url(thumb_url)
                if normalized_url not in image_urls:
                    image_urls.append(normalized_url)

        # Look for high-res image URLs in JavaScript
        high_res_urls = self._extract_high_res_images(context.html)
        for url in high_res_urls:
            if url not in image_urls:
                image_urls.append(url)

        return {
            'image_urls': image_urls,
            'image_count': len(image_urls),
            'main_image_path': image_urls[0] if image_urls else None
        }

    async def _extract_seller_info(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract seller information"""

        data = {}

        # Seller name
        seller_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('seller_name')
        )
        data['seller_id'] = self._clean_text(
            seller_elem.get_text() if seller_elem else "")

        # Feedback score
        feedback_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('feedback_score')
        )

        if feedback_elem:
            feedback_text = self._clean_text(feedback_elem.get_text())
            data['seller_feedback_count'] = self._extract_number(feedback_text)

        # Feedback percentage
        percentage_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('feedback_percentage')
        )

        if percentage_elem:
            percentage_text = self._clean_text(percentage_elem.get_text())
            data['seller_rating'] = self._extract_percentage(percentage_text)

        return data

    async def _extract_shipping_info(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract shipping information"""

        data = {}

        # Shipping cost
        shipping_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('shipping_cost')
        )

        if shipping_elem:
            shipping_text = self._clean_text(shipping_elem.get_text())
            data['shipping_cost'] = self._extract_price_value(shipping_text)

        # Ships from
        ships_from_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('ships_from')
        )
        data['ships_from'] = self._clean_text(
            ships_from_elem.get_text() if ships_from_elem else "")

        return data

    async def _extract_specifications(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract item specifications from the specifics table"""

        specifications = []

        # Find item specifics section
        specifics_elem = self._find_by_selectors(
            context.soup,
            self.selector_manager.get_all_selectors('item_specifics')
        )

        if specifics_elem:
            # Extract table rows
            rows = specifics_elem.find_all(
                'tr') or specifics_elem.find_all('div', class_='attrLabels')

            for row in rows:
                try:
                    # Try different table structures
                    cells = row.find_all(['td', 'th', 'div'])
                    if len(cells) >= 2:
                        attr_name = self._clean_text(cells[0].get_text())
                        attr_value = self._clean_text(cells[1].get_text())

                        if attr_name and attr_value:
                            spec = {
                                'spec_id': str(uuid.uuid4()),
                                'listing_id': context.listing_id,
                                'attribute_name': attr_name,
                                'attribute_value': attr_value,
                                'confidence_score': 0.9  # High confidence for structured data
                            }
                            specifications.append(spec)

                except Exception as e:
                    self.logger.debug(
                        f"Failed to extract specification row: {e}")
                    continue

        return {'specifications': specifications}

    async def _extract_metadata(self, context: ExtractionContext) -> Dict[str, Any]:
        """Extract additional metadata"""

        return {
            'scraped_at': context.extraction_time,
            'listing_url': context.url,
            'metadata': {
                'extraction_method': 'individual_listing',
                'html_size': len(context.html),
                'extraction_timestamp': context.extraction_time.isoformat()
            }
        }

    def _find_by_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> Optional[Tag]:
        """Find element using multiple selectors as fallbacks"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    return element
            except Exception:
                continue
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())

        # Remove common eBay artifacts
        cleaned = re.sub(r'\bSee original listing\b', '',
                         cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\beBay item number.*$', '',
                         cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _extract_price_value(self, price_text: str) -> float:
        """Extract numeric price from text"""
        if not price_text:
            return 0.0

        for pattern in self.price_patterns:
            match = re.search(pattern, price_text, re.IGNORECASE)
            if match:
                # Extract numbers from match
                number_text = re.sub(r'[^\d.,]', '', match.group())
                try:
                    # Handle comma as thousands separator
                    if ',' in number_text and '.' in number_text:
                        number_text = number_text.replace(',', '')
                    elif ',' in number_text:
                        # Could be thousands separator or decimal
                        if number_text.count(',') == 1 and len(number_text.split(',')[1]) <= 2:
                            number_text = number_text.replace(',', '.')
                        else:
                            number_text = number_text.replace(',', '')

                    return float(number_text)
                except ValueError:
                    continue

        return 0.0

    def _extract_currency(self, price_text: str) -> str:
        """Extract currency from price text"""
        if not price_text:
            return "USD"

        currency_patterns = {
            'USD': [r'\$', r'USD', r'US\$'],
            'EUR': [r'€', r'EUR'],
            'GBP': [r'£', r'GBP'],
            'CAD': [r'CAD', r'C\$'],
            'AUD': [r'AUD', r'A\$']
        }

        for currency, patterns in currency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, price_text, re.IGNORECASE):
                    return currency

        return "USD"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract product description"""

        # Try to find description iframe first
        iframe = soup.find('iframe', {'id': 'desc_ifr'})
        if iframe:
            # Description is in iframe - we'd need to load it separately
            # For now, fall back to other methods
            pass

        # Look for description in various containers
        desc_selectors = [
            'div[data-testid="x-item-description-label"]',
            '.viTabs div[data-testid="readMore"]',
            '.u-flL.condText',
            '#viTabs_0_pd',
            '.itemAttr'
        ]

        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem:
                return self._clean_text(elem.get_text())

        return ""

    def _extract_features(self, soup: BeautifulSoup, description: str) -> List[str]:
        """Extract product features"""
        features = []

        # Extract from bullet points
        bullet_selectors = ['ul li', '.itemAttr li', 'div.feature']

        for selector in bullet_selectors:
            elements = soup.select(selector)
            for elem in elements[:10]:  # Limit to first 10
                feature = self._clean_text(elem.get_text())
                if feature and len(feature) > 10:  # Meaningful features
                    features.append(feature)

        # Extract from description using patterns
        if description:
            # Look for bullet points in description
            bullet_matches = re.findall(r'[•*-]\s*([^•*-\n]+)', description)
            features.extend([f.strip()
                            for f in bullet_matches if len(f.strip()) > 10])

        # Remove duplicates and return
        return list(dict.fromkeys(features))[:20]  # Limit to 20 features

    def _detect_primary_material(self, text: str) -> JewelryMaterial:
        """Detect primary jewelry material"""

        for material, patterns in self.material_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return material

        return JewelryMaterial.UNKNOWN

    def _detect_all_materials(self, text: str) -> List[str]:
        """Detect all mentioned materials"""
        materials = []

        for material, patterns in self.material_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    materials.append(material.value)
                    break

        return materials

    def _detect_primary_gemstone(self, text: str) -> Optional[str]:
        """Detect primary gemstone"""

        for pattern in self.gemstone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().lower()

        return None

    def _categorize_jewelry(self, title: str) -> JewelryCategory:
        """Categorize jewelry based on title"""
        title_lower = title.lower()

        category_keywords = {
            JewelryCategory.RINGS: ['ring', 'wedding', 'engagement', 'band'],
            JewelryCategory.NECKLACES: ['necklace', 'chain', 'pendant'],
            JewelryCategory.EARRINGS: ['earring', 'stud', 'hoop', 'drop'],
            JewelryCategory.BRACELETS: ['bracelet', 'bangle', 'cuff'],
            JewelryCategory.WATCHES: ['watch', 'timepiece'],
            JewelryCategory.BROOCHES: ['brooch', 'pin'],
            JewelryCategory.CHAINS: ['chain'],
            JewelryCategory.PENDANTS: ['pendant']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return category

        return JewelryCategory.OTHER

    def _extract_size(self, text: str, category: JewelryCategory) -> Optional[str]:
        """Extract size based on jewelry category"""

        category_mapping = {
            JewelryCategory.RINGS: 'ring',
            JewelryCategory.NECKLACES: 'chain',
            JewelryCategory.BRACELETS: 'bracelet',
            JewelryCategory.CHAINS: 'chain'
        }

        pattern_key = category_mapping.get(category)
        if not pattern_key or pattern_key not in self.size_patterns:
            return None

        for pattern in self.size_patterns[pattern_key]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_stone_attributes(self, text: str) -> Dict[str, Optional[str]]:
        """Extract stone-specific attributes"""

        attributes = {
            'stone_color': None,
            'stone_clarity': None,
            'stone_cut': None,
            'stone_carat': None
        }

        # Color patterns
        color_pattern = r'\b(white|yellow|blue|red|green|pink|black|clear|colorless)\s*(?:color|colored)?\b'
        color_match = re.search(color_pattern, text, re.IGNORECASE)
        if color_match:
            attributes['stone_color'] = color_match.group(1).lower()

        # Clarity patterns
        clarity_pattern = r'\b(FL|IF|VVS1|VVS2|VS1|VS2|SI1|SI2|I1|I2|I3|flawless|eye\s*clean)\b'
        clarity_match = re.search(clarity_pattern, text, re.IGNORECASE)
        if clarity_match:
            attributes['stone_clarity'] = clarity_match.group(1).upper()

        # Cut patterns
        cut_pattern = r'\b(round|princess|emerald|oval|marquise|pear|cushion|heart|radiant|brilliant)\s*(?:cut)?\b'
        cut_match = re.search(cut_pattern, text, re.IGNORECASE)
        if cut_match:
            attributes['stone_cut'] = cut_match.group(1).lower()

        # Carat patterns
        carat_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:ct|carat|carats)\b'
        carat_match = re.search(carat_pattern, text, re.IGNORECASE)
        if carat_match:
            attributes['stone_carat'] = carat_match.group(1)

        return attributes

    def _normalize_image_url(self, url: str) -> str:
        """Normalize and enhance image URL"""
        if not url:
            return ""

        # Make absolute URL
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            url = 'https://www.ebay.com' + url

        # Try to get higher quality version
        # eBay image URL patterns for higher quality
        if 'ebayimg.com' in url:
            # Replace size parameter for higher quality
            url = re.sub(r's-l\d+', 's-l1600', url)  # Large size
            url = re.sub(r'\$_\d+\.JPG', '$_57.JPG', url)  # Higher quality

        return url

    def _extract_high_res_images(self, html: str) -> List[str]:
        """Extract high-resolution image URLs from JavaScript"""

        urls = []

        # Look for image URL patterns in JavaScript
        js_patterns = [
            r'"originalImg":"([^"]+)"',
            r'"imageUrls":\[([^\]]+)\]',
            r'"mainImgUrl":"([^"]+)"',
            r'imgURL:\s*"([^"]+)"'
        ]

        for pattern in js_patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                if isinstance(match, str) and 'ebayimg.com' in match:
                    # Unescape URL
                    url = match.replace('\\/', '/')
                    urls.append(self._normalize_image_url(url))

        return urls

    def _extract_item_number(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract eBay item number"""

        # Look for item number in various places
        patterns = [
            r'Item number[:\s]*(\d+)',
            r'eBay item number[:\s]*(\d+)',
            r'#(\d{12,})'  # eBay item numbers are typically 12+ digits
        ]

        text = soup.get_text()
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _determine_listing_type(self, soup: BeautifulSoup) -> str:
        """Determine listing type (auction, BIN, etc.)"""

        text = soup.get_text().lower()

        if 'buy it now' in text or 'bin' in text:
            return "Buy It Now"
        elif 'auction' in text or 'bid' in text:
            return "Auction"
        elif 'best offer' in text:
            return "Best Offer"
        else:
            return "Unknown"

    def _determine_listing_status(self, soup: BeautifulSoup) -> ListingStatus:
        """Determine current listing status"""

        text = soup.get_text().lower()

        if any(word in text for word in ['sold', 'ended', 'no longer available']):
            return ListingStatus.SOLD
        elif any(word in text for word in ['active', 'available', 'buy it now']):
            return ListingStatus.ACTIVE
        else:
            return ListingStatus.UNKNOWN

    def _extract_availability(self, soup: BeautifulSoup) -> str:
        """Extract availability information"""

        # Look for availability indicators
        availability_selectors = [
            '.notifTxt',
            '.msgPad',
            '.vim x-quantity-availability'
        ]

        for selector in availability_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = self._clean_text(elem.get_text())
                if text:
                    return text

        return "Available"

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text"""

        if not text:
            return None

        match = re.search(r'(\d+)', text.replace(',', ''))
        return int(match.group(1)) if match else None

    def _extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage from text"""

        if not text:
            return None

        match = re.search(r'(\d+(?:\.\d+)?)%', text)
        return float(match.group(1)) if match else None

    def _extract_dimensions(self, text: str) -> Optional[str]:
        """Extract dimensions from text"""

        dimension_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*(?:x|×)?\s*(\d+(?:\.\d+)?)?\s*(?:inches?|in|cm|mm)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:L|length)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*(?:W|width)\b'
        ]

        for pattern in dimension_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()

        return None

    def _extract_weight(self, text: str) -> Optional[str]:
        """Extract weight from text"""

        weight_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(?:grams?|g|oz|ounces?|lbs?|pounds?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:dwt|pennyweight)\b'
        ]

        for pattern in weight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()

        return None

    def _create_jewelry_listing(self, data: Dict[str, Any], context: ExtractionContext) -> JewelryListing:
        """Create JewelryListing object from extracted data"""

        # Create listing with extracted data
        listing = JewelryListing(
            id=context.listing_id,
            title=data.get('title', ''),
            price=data.get('price', 0.0),
            currency=data.get('currency', 'USD'),
            condition=data.get('condition', 'Unknown'),
            seller_id=data.get('seller_id', ''),
            listing_url=context.url,
            category=data.get('category', JewelryCategory.OTHER),
            material=data.get('material', JewelryMaterial.UNKNOWN),
            gemstone=data.get('gemstone'),
            size=data.get('size'),
            weight=data.get('weight'),
            brand=data.get('brand'),
            image_urls=data.get('image_urls', []),
            main_image_path=data.get('main_image_path'),
            scraped_at=data.get('scraped_at', datetime.now()),
            original_price=data.get('original_price'),
            seller_rating=data.get('seller_rating'),
            seller_feedback_count=data.get('seller_feedback_count'),
            materials=data.get('materials', []),
            dimensions=data.get('dimensions'),
            stone_color=data.get('stone_color'),
            stone_clarity=data.get('stone_clarity'),
            stone_cut=data.get('stone_cut'),
            stone_carat=data.get('stone_carat'),
            description=data.get('description', ''),
            features=data.get('features', []),
            item_number=data.get('item_number'),
            listing_type=data.get('listing_type'),
            listing_status=data.get('listing_status', ListingStatus.UNKNOWN),
            shipping_cost=data.get('shipping_cost'),
            ships_from=data.get('ships_from'),
            availability=data.get('availability', 'Available'),
            image_count=data.get('image_count', 0),
            description_length=data.get('description_length', 0),
            metadata=data.get('metadata', {}),
            raw_data={
                'extraction_context': {
                    'url': context.url,
                    'extraction_time': context.extraction_time.isoformat(),
                    'html_size': len(context.html)
                },
                'specifications': data.get('specifications', [])
            }
        )

        return listing
