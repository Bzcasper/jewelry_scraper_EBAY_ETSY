"""
Enhanced Jewelry Scraper Database Models
Pydantic models with advanced validation and enhanced features for jewelry listings system.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import re
import hashlib
from urllib.parse import urlparse
from pathlib import Path


class JewelryCategory(str, Enum):
    """Enhanced jewelry category classification"""
    RINGS = "rings"
    NECKLACES = "necklaces"  
    EARRINGS = "earrings"
    BRACELETS = "bracelets"
    WATCHES = "watches"
    BROOCHES = "brooches"
    ANKLETS = "anklets"
    PENDANTS = "pendants"
    CHAINS = "chains"
    SETS = "sets"
    CHARMS = "charms"
    CUFFLINKS = "cufflinks"
    PINS = "pins"
    TIARAS = "tiaras"
    BODY_JEWELRY = "body_jewelry"
    VINTAGE = "vintage"
    ANTIQUE = "antique"
    LUXURY = "luxury"
    CUSTOM = "custom"
    OTHER = "other"


class JewelryMaterial(str, Enum):
    """Enhanced jewelry material types with precious metals and alloys"""
    # Precious metals
    GOLD_24K = "gold_24k"
    GOLD_18K = "gold_18k"
    GOLD_14K = "gold_14k"
    GOLD_10K = "gold_10k"
    GOLD_9K = "gold_9k"
    WHITE_GOLD = "white_gold"
    ROSE_GOLD = "rose_gold"
    YELLOW_GOLD = "yellow_gold"
    
    # Silver
    STERLING_SILVER = "sterling_silver"
    SILVER_925 = "silver_925"
    SILVER_950 = "silver_950"
    SILVER_PLATED = "silver_plated"
    
    # Platinum group
    PLATINUM = "platinum"
    PALLADIUM = "palladium"
    RHODIUM = "rhodium"
    
    # Other metals
    TITANIUM = "titanium"
    STAINLESS_STEEL = "stainless_steel"
    SURGICAL_STEEL = "surgical_steel"
    COPPER = "copper"
    BRASS = "brass"
    BRONZE = "bronze"
    ALUMINUM = "aluminum"
    
    # Non-metals
    LEATHER = "leather"
    FABRIC = "fabric"
    RUBBER = "rubber"
    PLASTIC = "plastic"
    RESIN = "resin"
    CERAMIC = "ceramic"
    GLASS = "glass"
    WOOD = "wood"
    BONE = "bone"
    SHELL = "shell"
    
    # Combinations
    MIXED_METALS = "mixed_metals"
    GOLD_FILLED = "gold_filled"
    VERMEIL = "vermeil"
    
    # Unknown/Other
    UNKNOWN = "unknown"
    OTHER = "other"


class ListingStatus(str, Enum):
    """eBay listing status with additional states"""
    ACTIVE = "active"
    SOLD = "sold"
    ENDED = "ended"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RELISTED = "relisted"
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    UNKNOWN = "unknown"


class ScrapingStatus(str, Enum):
    """Enhanced scraping session status"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"
    RETRYING = "retrying"


class ImageType(str, Enum):
    """Enhanced image classification types"""
    MAIN = "main"
    GALLERY = "gallery"
    DETAIL = "detail"
    CERTIFICATE = "certificate"
    APPRAISAL = "appraisal"
    PACKAGING = "packaging"
    COMPARISON = "comparison"
    LIFESTYLE = "lifestyle"
    SCALE = "scale"
    MACRO = "macro"
    ANGLE_VIEW = "angle_view"
    BACK_VIEW = "back_view"
    SIDE_VIEW = "side_view"
    CLASP = "clasp"
    HALLMARK = "hallmark"
    DEFECT = "defect"
    REPAIR = "repair"


class DataQuality(str, Enum):
    """Data quality classification"""
    EXCELLENT = "excellent"  # 0.9-1.0
    GOOD = "good"           # 0.7-0.89
    FAIR = "fair"           # 0.5-0.69
    POOR = "poor"           # 0.3-0.49
    INVALID = "invalid"     # 0.0-0.29


class JewelryListing(BaseModel):
    """
    Enhanced jewelry listing data model with comprehensive validation and quality scoring
    """
    
    # === CORE REQUIRED FIELDS ===
    id: str = Field(..., description="Unique listing identifier", min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=500, description="Listing title")
    price: float = Field(..., gt=0, description="Current price (must be positive)")
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$", description="ISO currency code")
    condition: str = Field(..., min_length=1, max_length=50, description="Item condition")
    
    # === EBAY SPECIFIC FIELDS ===
    seller_id: str = Field(..., min_length=1, max_length=100, description="eBay seller username/ID")
    listing_url: str = Field(..., description="Full eBay listing URL")
    ebay_item_id: Optional[str] = Field(None, description="eBay internal item ID")
    end_time: Optional[datetime] = Field(None, description="Listing end time")
    start_time: Optional[datetime] = Field(None, description="Listing start time")
    shipping_cost: Optional[float] = Field(None, ge=0, description="Shipping cost (non-negative)")
    
    # === JEWELRY SPECIFIC REQUIRED FIELDS ===
    category: JewelryCategory = Field(..., description="Primary jewelry category")
    material: JewelryMaterial = Field(..., description="Primary jewelry material")
    
    # === ENHANCED JEWELRY ATTRIBUTES ===
    gemstone: Optional[str] = Field(None, max_length=100, description="Primary gemstone type")
    gemstone_treatment: Optional[str] = Field(None, description="Gemstone treatment information")
    size: Optional[str] = Field(None, max_length=50, description="Size information")
    ring_size: Optional[str] = Field(None, description="Ring size (if applicable)")
    length: Optional[str] = Field(None, description="Length measurement")
    width: Optional[str] = Field(None, description="Width measurement")
    thickness: Optional[str] = Field(None, description="Thickness measurement")
    weight: Optional[str] = Field(None, max_length=50, description="Weight information")
    carat_weight: Optional[float] = Field(None, ge=0, description="Total carat weight")
    brand: Optional[str] = Field(None, max_length=100, description="Brand name")
    manufacturer: Optional[str] = Field(None, max_length=100, description="Manufacturer name")
    designer: Optional[str] = Field(None, max_length=100, description="Designer name")
    collection: Optional[str] = Field(None, max_length=100, description="Collection name")
    style: Optional[str] = Field(None, max_length=100, description="Style description")
    era: Optional[str] = Field(None, description="Era/period (vintage, antique, etc.)")
    country_origin: Optional[str] = Field(None, description="Country of origin")
    
    # === ENHANCED GEMSTONE DETAILS ===
    main_stone_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Detailed main stone information")
    stone_color: Optional[str] = Field(None, max_length=50, description="Primary stone color")
    stone_clarity: Optional[str] = Field(None, max_length=50, description="Stone clarity grade")
    stone_cut: Optional[str] = Field(None, max_length=50, description="Stone cut type")
    stone_shape: Optional[str] = Field(None, max_length=50, description="Stone shape")
    stone_carat: Optional[str] = Field(None, max_length=20, description="Stone carat weight")
    stone_certification: Optional[str] = Field(None, description="Certification information")
    accent_stones: List[Dict[str, Any]] = Field(default_factory=list, description="Accent/secondary stones details")
    
    # === MATERIALS AND COMPOSITION ===
    materials: List[str] = Field(default_factory=list, description="All materials mentioned")
    metal_purity: Optional[str] = Field(None, description="Metal purity/karat")
    metal_stamp: Optional[str] = Field(None, description="Metal stamp/hallmark")
    plating: Optional[str] = Field(None, description="Plating information")
    finish: Optional[str] = Field(None, description="Surface finish")
    
    # === IMAGES AND MEDIA ===
    image_urls: List[str] = Field(default_factory=list, description="List of all image URLs")
    main_image_url: Optional[str] = Field(None, description="Primary image URL")
    main_image_path: Optional[str] = Field(None, description="Path to main/primary image")
    image_count: int = Field(default=0, ge=0, description="Number of images")
    has_video: bool = Field(default=False, description="Whether listing has video")
    video_urls: List[str] = Field(default_factory=list, description="Video URLs")
    
    # === DESCRIPTION AND CONTENT ===
    description: Optional[str] = Field(None, max_length=50000, description="Full item description")
    description_html: Optional[str] = Field(None, description="HTML description")
    features: List[str] = Field(default_factory=list, description="Key features")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Technical specifications")
    tags: List[str] = Field(default_factory=list, description="Generated tags")
    keywords: List[str] = Field(default_factory=list, description="SEO keywords")
    
    # === ENHANCED EBAY DETAILS ===
    item_number: Optional[str] = Field(None, max_length=50, description="eBay item number")
    listing_type: Optional[str] = Field(None, max_length=50, description="Auction, Buy It Now, etc.")
    listing_format: Optional[str] = Field(None, description="Fixed price, auction, etc.")
    listing_status: ListingStatus = Field(default=ListingStatus.UNKNOWN, description="Current listing status")
    listing_duration: Optional[str] = Field(None, description="Listing duration")
    
    # === ENGAGEMENT METRICS ===
    watchers: Optional[int] = Field(None, ge=0, description="Number of watchers")
    views: Optional[int] = Field(None, ge=0, description="Number of views")
    page_views: Optional[int] = Field(None, ge=0, description="Page view count")
    bids: Optional[int] = Field(None, ge=0, description="Number of bids")
    bid_count: Optional[int] = Field(None, ge=0, description="Current bid count")
    max_bid: Optional[float] = Field(None, ge=0, description="Highest bid amount")
    reserve_met: Optional[bool] = Field(None, description="Reserve price met")
    time_left: Optional[str] = Field(None, description="Time remaining")
    
    # === SELLER INFORMATION ===
    seller_rating: Optional[float] = Field(None, ge=0, le=100, description="Seller feedback score")
    seller_feedback_count: Optional[int] = Field(None, ge=0, description="Number of feedback items")
    seller_feedback_percentage: Optional[float] = Field(None, ge=0, le=100, description="Positive feedback percentage")
    seller_location: Optional[str] = Field(None, description="Seller location")
    seller_store_name: Optional[str] = Field(None, description="eBay store name")
    seller_level: Optional[str] = Field(None, description="Seller level (Top Rated, etc.)")
    
    # === PRICING INFORMATION ===
    original_price: Optional[float] = Field(None, gt=0, description="Original/retail price")
    buy_it_now_price: Optional[float] = Field(None, gt=0, description="Buy It Now price")
    starting_bid: Optional[float] = Field(None, gt=0, description="Starting bid amount")
    reserve_price: Optional[float] = Field(None, gt=0, description="Reserve price")
    best_offer: Optional[bool] = Field(None, description="Best offer available")
    price_drop: Optional[float] = Field(None, description="Price drop amount")
    discount_percentage: Optional[float] = Field(None, ge=0, le=100, description="Discount percentage")
    
    # === AVAILABILITY AND INVENTORY ===
    availability: Optional[str] = Field(None, max_length=100, description="Availability status")
    quantity_available: Optional[int] = Field(None, ge=0, description="Available quantity")
    quantity_sold: Optional[int] = Field(None, ge=0, description="Quantity sold")
    inventory_status: Optional[str] = Field(None, description="Inventory status")
    
    # === SHIPPING AND LOGISTICS ===
    ships_from: Optional[str] = Field(None, max_length=100, description="Shipping location")
    ships_to: Optional[str] = Field(None, description="Shipping destinations")
    shipping_methods: List[str] = Field(default_factory=list, description="Available shipping methods")
    expedited_shipping: Optional[bool] = Field(None, description="Expedited shipping available")
    international_shipping: Optional[bool] = Field(None, description="International shipping available")
    handling_time: Optional[str] = Field(None, description="Handling time")
    return_policy: Optional[str] = Field(None, description="Return policy")
    return_period: Optional[str] = Field(None, description="Return period")
    
    # === CATEGORIZATION ===
    subcategory: Optional[str] = Field(None, max_length=100, description="Specific subcategory")
    ebay_category: Optional[str] = Field(None, description="eBay category path")
    custom_categories: List[str] = Field(default_factory=list, description="Custom categorization")
    
    # === QUALITY METRICS ===
    description_length: int = Field(default=0, ge=0, description="Description character count")
    title_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Title quality score")
    description_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Description quality score")
    image_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Image quality score")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall data quality score")
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data completeness score")
    
    # === TIMESTAMPS ===
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Timestamp when scraped")
    listing_date: Optional[datetime] = Field(None, description="When listing was posted")
    last_seen: Optional[datetime] = Field(None, description="Last time listing was seen active")
    
    # === VALIDATION AND METADATA ===
    is_validated: bool = Field(default=False, description="Data validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    validation_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation confidence score")
    data_source: str = Field(default="ebay", description="Data source identifier")
    scraper_version: Optional[str] = Field(None, description="Scraper version used")
    
    # === ADVANCED METADATA ===
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw scraped data")
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction process metadata")
    processing_flags: Dict[str, bool] = Field(default_factory=dict, description="Processing status flags")
    
    # === HASH AND DEDUPLICATION ===
    content_hash: Optional[str] = Field(None, description="Content hash for deduplication")
    listing_hash: Optional[str] = Field(None, description="Listing-specific hash")
    
    # === ADVANCED PYDANTIC VALIDATORS ===
    
    @validator('listing_url')
    def validate_listing_url(cls, v):
        """Enhanced eBay listing URL validation"""
        if not v:
            raise ValueError('Listing URL is required')
        
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Invalid URL format')
        except Exception:
            raise ValueError('Invalid URL format')
        
        # Check for eBay domain
        if not any(domain in v.lower() for domain in ['ebay.com', 'ebay.co.uk', 'ebay.de', 'ebay.fr', 'ebay.au', 'ebay.ca']):
            raise ValueError('URL must be from eBay domain')
        
        return v
    
    @validator('image_urls')
    def validate_image_urls(cls, v):
        """Enhanced image URL validation"""
        if not v:
            return v
        
        valid_urls = []
        for url in v:
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    # Check for common image extensions
                    path_lower = parsed.path.lower()
                    if any(ext in path_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']) or 'image' in url.lower():
                        valid_urls.append(url)
            except Exception:
                continue
        
        return valid_urls
    
    @validator('price', 'original_price', 'buy_it_now_price', 'starting_bid', 'reserve_price')
    def validate_prices(cls, v):
        """Validate price fields"""
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @validator('seller_id')
    def validate_seller_id(cls, v):
        """Enhanced seller ID validation"""
        if not v or not v.strip():
            raise ValueError('Seller ID cannot be empty')
        
        # Clean seller ID
        cleaned = v.strip()
        
        # Check length
        if len(cleaned) < 3:
            raise ValueError('Seller ID must be at least 3 characters')
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[a-zA-Z0-9_-]+$', cleaned):
            raise ValueError('Seller ID contains invalid characters')
        
        return cleaned
    
    @validator('title')
    def validate_title(cls, v):
        """Enhanced title validation and cleaning"""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        
        # Clean title
        cleaned = v.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
        cleaned = re.sub(r'[^\w\s\-\.,:;!?()&\'"]+', '', cleaned)  # Remove unusual characters
        
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + '...'
        
        return cleaned
    
    @validator('condition')
    def validate_condition(cls, v):
        """Enhanced condition validation and standardization"""
        if not v or not v.strip():
            raise ValueError('Condition cannot be empty')
        
        # Standardize common condition values
        condition_map = {
            'new': 'New',
            'new with tags': 'New with tags',
            'new without tags': 'New without tags',
            'new with defects': 'New with defects',
            'manufacturer refurbished': 'Manufacturer refurbished',
            'seller refurbished': 'Seller refurbished',
            'used': 'Used',
            'very good': 'Very Good',
            'good': 'Good',
            'acceptable': 'Acceptable',
            'for parts': 'For parts or not working',
            'for parts or not working': 'For parts or not working'
        }
        
        cleaned = v.strip().lower()
        return condition_map.get(cleaned, v.strip())
    
    @validator('currency')
    def validate_currency(cls, v):
        """Enhanced currency code validation"""
        if not re.match(r'^[A-Z]{3}$', v):
            raise ValueError('Currency must be a 3-letter ISO code (e.g., USD, EUR, GBP)')
        
        # Check against common currency codes
        valid_currencies = {
            'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'CNY', 'INR', 'KRW',
            'SGD', 'HKD', 'NOK', 'SEK', 'DKK', 'PLN', 'CZK', 'HUF', 'ILS', 'MXN'
        }
        
        if v not in valid_currencies:
            # Still allow but log warning
            pass
        
        return v.upper()
    
    def calculate_enhanced_quality_score(self) -> float:
        """
        Advanced data quality scoring with weighted categories
        """
        score = 0.0
        
        # Core fields (30% weight) - absolutely essential
        core_fields = ['id', 'title', 'price', 'currency', 'condition', 'seller_id', 'listing_url', 'category', 'material']
        core_filled = sum(1 for field in core_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (core_filled / len(core_fields)) * 0.30
        
        # Important fields (25% weight) - significantly improve data value
        important_fields = ['brand', 'gemstone', 'size', 'weight', 'description', 'image_urls', 'shipping_cost', 'seller_rating']
        important_filled = sum(1 for field in important_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (important_filled / len(important_fields)) * 0.25
        
        # Enhancement fields (20% weight) - add significant value
        enhancement_fields = ['stone_color', 'stone_clarity', 'stone_cut', 'features', 'specifications', 'materials', 'metal_purity']
        enhancement_filled = sum(1 for field in enhancement_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (enhancement_filled / len(enhancement_fields)) * 0.20
        
        # Engagement fields (10% weight) - market validation
        engagement_fields = ['watchers', 'views', 'bids', 'seller_feedback_count']
        engagement_filled = sum(1 for field in engagement_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (engagement_filled / len(engagement_fields)) * 0.10
        
        # Quality bonuses (15% weight)
        quality_score = 0.0
        
        # Image quality bonus (max 40% of quality score)
        if self.image_count > 0:
            image_bonus = min(self.image_count / 15, 0.4)  # Peak at 15 images
            quality_score += image_bonus
        
        # Description quality bonus (max 30% of quality score)
        if self.description_length > 50:
            desc_bonus = min(self.description_length / 2000, 0.3)  # Peak at 2000 chars
            quality_score += desc_bonus
        
        # Completeness bonus (max 30% of quality score)
        total_possible_fields = 50  # Approximate total meaningful fields
        filled_fields = sum(1 for field in self.__fields__ if self._is_field_valuable(getattr(self, field, None)))
        completeness_bonus = min(filled_fields / total_possible_fields, 0.3)
        quality_score += completeness_bonus
        
        score += min(quality_score, 1.0) * 0.15
        
        return round(min(score, 1.0), 3)
    
    def _is_field_valuable(self, value) -> bool:
        """Enhanced field value assessment"""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip()) and value.strip().lower() not in ['unknown', 'n/a', 'null', 'none', '']
        if isinstance(value, (list, dict)):
            return len(value) > 0
        if isinstance(value, (int, float)):
            return value > 0
        return bool(value)
    
    def get_quality_classification(self) -> DataQuality:
        """Get quality classification based on score"""
        score = self.data_quality_score
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        elif score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    def generate_content_hash(self) -> str:
        """Generate content hash for deduplication"""
        # Use key fields for hash generation
        hash_content = f"{self.title}|{self.price}|{self.seller_id}|{self.category}|{self.material}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def update_quality_score(self):
        """Update quality score and related fields"""
        self.data_quality_score = self.calculate_enhanced_quality_score()
        self.updated_at = datetime.now()
        self.content_hash = self.generate_content_hash()
        
        # Auto-validate if quality score is high enough
        if self.data_quality_score >= 0.6:
            self.is_validated = True
            self.validation_errors = []
            self.validation_score = self.data_quality_score
    
    def validate_for_database(self) -> bool:
        """Enhanced database validation"""
        errors = []
        
        # Required field validation
        required_validations = [
            (self.id, "ID is required"),
            (self.title, "Title is required"),
            (self.price and self.price > 0, "Valid price is required"),
            (self.seller_id, "Seller ID is required"),
            (self.listing_url, "Listing URL is required"),
            (self.category, "Category is required"),
            (self.material, "Material is required")
        ]
        
        for condition, error_msg in required_validations:
            if not condition:
                errors.append(error_msg)
        
        # Data quality validation
        if self.data_quality_score < 0.3:
            errors.append("Data quality score too low")
        
        # URL validation
        if self.listing_url:
            try:
                parsed = urlparse(self.listing_url)
                if not parsed.scheme or not parsed.netloc:
                    errors.append("Invalid listing URL format")
            except:
                errors.append("Invalid listing URL")
        
        # Price validation
        if self.price and self.original_price and self.price > self.original_price * 2:
            errors.append("Current price seems unusually high compared to original price")
        
        self.validation_errors = errors
        self.is_validated = len(errors) == 0
        self.validation_score = 1.0 - (len(errors) / 10)  # Penalty for each error
        
        return self.is_validated
    
    def to_dict(self, include_metadata: bool = False, include_raw_data: bool = False) -> Dict[str, Any]:
        """Enhanced dictionary conversion with options"""
        data = self.dict()
        
        if not include_metadata:
            # Remove internal fields for cleaner output
            fields_to_remove = ['raw_data', 'validation_errors', 'extraction_metadata', 'processing_flags']
            for field in fields_to_remove:
                data.pop(field, None)
        
        if not include_raw_data:
            data.pop('raw_data', None)
        
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced listing summary with quality indicators"""
        return {
            'id': self.id,
            'title': self.title[:100] + '...' if len(self.title) > 100 else self.title,
            'price': f"{self.currency} {self.price:.2f}",
            'original_price': f"{self.currency} {self.original_price:.2f}" if self.original_price else None,
            'category': self.category.value,
            'material': self.material.value,
            'brand': self.brand or 'Unknown',
            'condition': self.condition,
            'seller': self.seller_id,
            'seller_rating': f"{self.seller_rating:.1f}%" if self.seller_rating else None,
            'images': len(self.image_urls),
            'watchers': self.watchers,
            'bids': self.bids,
            'quality_score': f"{self.data_quality_score:.1%}",
            'quality_class': self.get_quality_classification().value,
            'scraped': self.scraped_at.strftime('%Y-%m-%d %H:%M'),
            'validated': self.is_validated,
            'url': self.listing_url
        }


class JewelryImage(BaseModel):
    """
    Enhanced image metadata model with advanced processing information
    """
    
    # Primary identifiers
    image_id: str = Field(..., description="Unique image identifier")
    listing_id: str = Field(..., description="Associated listing ID")
    
    # Image source and storage
    original_url: str = Field(..., description="Original image URL")
    local_path: Optional[str] = Field(None, description="Local file path")
    filename: Optional[str] = Field(None, description="Generated filename")
    
    # Enhanced image classification
    image_type: ImageType = Field(default=ImageType.GALLERY, description="Image type/purpose")
    sequence_order: int = Field(default=0, ge=0, description="Order in listing gallery")
    is_primary: bool = Field(default=False, description="Primary/main image flag")
    
    # Image properties
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    width: Optional[int] = Field(None, ge=0, description="Image width in pixels")
    height: Optional[int] = Field(None, ge=0, description="Image height in pixels")
    format: Optional[str] = Field(None, description="Image format (jpg, png, etc.)")
    color_mode: Optional[str] = Field(None, description="Color mode (RGB, CMYK, etc.)")
    bit_depth: Optional[int] = Field(None, description="Bit depth")
    
    # Enhanced quality and processing
    is_processed: bool = Field(default=False, description="Image processing status")
    is_optimized: bool = Field(default=False, description="Image optimization status")
    is_resized: bool = Field(default=False, description="Image resizing status")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Image quality assessment")
    resolution_score: Optional[float] = Field(None, ge=0, le=1, description="Resolution quality score")
    sharpness_score: Optional[float] = Field(None, ge=0, le=1, description="Image sharpness score")
    brightness_score: Optional[float] = Field(None, ge=0, le=1, description="Brightness quality score")
    
    # Enhanced content analysis
    contains_text: bool = Field(default=False, description="Whether image contains text")
    contains_watermark: bool = Field(default=False, description="Whether image contains watermark")
    contains_logo: bool = Field(default=False, description="Whether image contains logo")
    is_duplicate: bool = Field(default=False, description="Duplicate detection flag")
    similarity_hash: Optional[str] = Field(None, description="Perceptual hash for similarity")
    color_histogram: Optional[str] = Field(None, description="Color histogram data")
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant color values")
    
    # AI/ML analysis
    ai_tags: List[str] = Field(default_factory=list, description="AI-generated tags")
    ai_confidence: Optional[float] = Field(None, ge=0, le=1, description="AI analysis confidence")
    object_detection: List[Dict[str, Any]] = Field(default_factory=list, description="Detected objects")
    
    # Alt text and descriptions
    alt_text: Optional[str] = Field(None, max_length=500, description="Alt text from HTML")
    generated_description: Optional[str] = Field(None, max_length=1000, description="AI-generated description")
    manual_description: Optional[str] = Field(None, max_length=1000, description="Manual description")
    
    # Processing history
    processing_history: List[Dict[str, Any]] = Field(default_factory=list, description="Processing operations history")
    optimization_settings: Dict[str, Any] = Field(default_factory=dict, description="Optimization settings used")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation time")
    downloaded_at: Optional[datetime] = Field(None, description="Download timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    last_verified: Optional[datetime] = Field(None, description="Last verification timestamp")
    
    # Error handling
    download_attempts: int = Field(default=0, ge=0, description="Number of download attempts")
    download_errors: List[str] = Field(default_factory=list, description="Download error messages")
    processing_errors: List[str] = Field(default_factory=list, description="Processing error messages")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    exif_data: Dict[str, Any] = Field(default_factory=dict, description="EXIF data")
    
    @validator('quality_score', 'resolution_score', 'sharpness_score', 'brightness_score', 'ai_confidence')
    def validate_scores(cls, v):
        if v is not None:
            return max(0.0, min(1.0, v))
        return v
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall image quality score"""
        scores = [
            self.quality_score or 0.0,
            self.resolution_score or 0.0,
            self.sharpness_score or 0.0,
            self.brightness_score or 0.0
        ]
        
        # Weight by availability
        valid_scores = [s for s in scores if s > 0]
        if not valid_scores:
            return 0.0
        
        return sum(valid_scores) / len(valid_scores)


class JewelrySpecification(BaseModel):
    """
    Enhanced specifications model with standardization and confidence scoring
    """
    
    # Primary identifiers
    spec_id: str = Field(..., description="Unique specification ID")
    listing_id: str = Field(..., description="Associated listing ID")
    
    # Specification details
    attribute_name: str = Field(..., max_length=200, description="Specification attribute name")
    attribute_value: str = Field(..., max_length=1000, description="Specification value")
    attribute_category: Optional[str] = Field(None, max_length=100, description="Category of specification")
    attribute_type: Optional[str] = Field(None, description="Data type (string, numeric, boolean, etc.)")
    
    # Enhanced extraction information
    source_section: Optional[str] = Field(None, description="Where this was extracted from")
    extraction_method: Optional[str] = Field(None, description="Method used for extraction")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Extraction confidence")
    
    # Enhanced standardization
    standardized_name: Optional[str] = Field(None, description="Standardized attribute name")
    standardized_value: Optional[str] = Field(None, description="Standardized value")
    standardized_category: Optional[str] = Field(None, description="Standardized category")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    unit_standardized: Optional[str] = Field(None, description="Standardized unit")
    
    # Value parsing
    numeric_value: Optional[float] = Field(None, description="Parsed numeric value")
    boolean_value: Optional[bool] = Field(None, description="Parsed boolean value")
    date_value: Optional[datetime] = Field(None, description="Parsed date value")
    
    # Quality and validation
    is_verified: bool = Field(default=False, description="Manual verification status")
    is_standardized: bool = Field(default=False, description="Standardization status")
    validation_status: Optional[str] = Field(None, description="Validation status")
    quality_score: float = Field(default=0.0, ge=0, le=1, description="Overall quality score")
    
    # Relationships
    related_specs: List[str] = Field(default_factory=list, description="Related specification IDs")
    conflicts_with: List[str] = Field(default_factory=list, description="Conflicting specification IDs")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    verified_at: Optional[datetime] = Field(None, description="Verification timestamp")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence_score', 'quality_score')
    def validate_scores(cls, v):
        return max(0.0, min(1.0, v))


class ScrapingSession(BaseModel):
    """
    Enhanced scraping session model with comprehensive tracking and analytics
    """
    
    # Primary identifiers
    session_id: str = Field(..., description="Unique session identifier")
    session_name: Optional[str] = Field(None, max_length=200, description="Human-readable session name")
    parent_session_id: Optional[str] = Field(None, description="Parent session for sub-sessions")
    
    # Enhanced session configuration
    search_query: Optional[str] = Field(None, max_length=500, description="Search query used")
    search_filters: Dict[str, Any] = Field(default_factory=dict, description="Applied search filters")
    search_categories: List[str] = Field(default_factory=list, description="Target categories")
    max_pages: Optional[int] = Field(None, ge=1, description="Maximum pages to scrape")
    max_listings: Optional[int] = Field(None, ge=1, description="Maximum listings to scrape")
    target_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Target quality score threshold")
    
    # Enhanced session status
    status: ScrapingStatus = Field(default=ScrapingStatus.INITIALIZED, description="Current session status")
    progress_percentage: float = Field(default=0.0, ge=0, le=100, description="Progress percentage")
    current_phase: Optional[str] = Field(None, description="Current processing phase")
    
    # Enhanced statistics
    listings_found: int = Field(default=0, ge=0, description="Total listings discovered")
    listings_scraped: int = Field(default=0, ge=0, description="Successfully scraped listings")
    listings_failed: int = Field(default=0, ge=0, description="Failed scraping attempts")
    listings_skipped: int = Field(default=0, ge=0, description="Skipped listings")
    listings_duplicate: int = Field(default=0, ge=0, description="Duplicate listings found")
    
    # Image statistics
    images_found: int = Field(default=0, ge=0, description="Total images discovered")
    images_downloaded: int = Field(default=0, ge=0, description="Images successfully downloaded")
    images_failed: int = Field(default=0, ge=0, description="Failed image downloads")
    images_skipped: int = Field(default=0, ge=0, description="Skipped images")
    
    # Performance metrics
    pages_processed: int = Field(default=0, ge=0, description="Search result pages processed")
    requests_made: int = Field(default=0, ge=0, description="Total HTTP requests made")
    requests_successful: int = Field(default=0, ge=0, description="Successful HTTP requests")
    requests_failed: int = Field(default=0, ge=0, description="Failed HTTP requests")
    data_volume_mb: float = Field(default=0.0, ge=0, description="Total data downloaded (MB)")
    
    # Quality metrics
    average_quality_score: float = Field(default=0.0, ge=0, le=1, description="Average quality score")
    high_quality_count: int = Field(default=0, ge=0, description="High quality listings count")
    low_quality_count: int = Field(default=0, ge=0, description="Low quality listings count")
    
    # Timing information
    started_at: datetime = Field(default_factory=datetime.now, description="Session start time")
    completed_at: Optional[datetime] = Field(None, description="Session completion time")
    paused_at: Optional[datetime] = Field(None, description="Session pause time")
    resumed_at: Optional[datetime] = Field(None, description="Session resume time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Enhanced error handling
    error_count: int = Field(default=0, ge=0, description="Total errors encountered")
    warning_count: int = Field(default=0, ge=0, description="Total warnings")
    critical_error_count: int = Field(default=0, ge=0, description="Critical errors")
    last_error: Optional[str] = Field(None, description="Most recent error message")
    error_categories: Dict[str, int] = Field(default_factory=dict, description="Error count by category")
    retry_count: int = Field(default=0, ge=0, description="Number of retries performed")
    
    # Enhanced configuration
    user_agent: Optional[str] = Field(None, description="User agent used")
    proxy_used: Optional[str] = Field(None, description="Proxy configuration")
    rate_limit_delay: float = Field(default=1.0, ge=0, description="Rate limiting delay (seconds)")
    concurrent_requests: int = Field(default=1, ge=1, description="Concurrent request limit")
    timeout_seconds: int = Field(default=30, ge=1, description="Request timeout")
    
    # Enhanced output configuration
    export_formats: List[str] = Field(default_factory=list, description="Requested export formats")
    output_directory: Optional[str] = Field(None, description="Output directory path")
    backup_enabled: bool = Field(default=True, description="Backup enabled flag")
    compression_enabled: bool = Field(default=False, description="Compression enabled flag")
    
    # Advanced metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage statistics")
    
    # Session metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
    tags: List[str] = Field(default_factory=list, description="Session tags")
    notes: Optional[str] = Field(None, description="Session notes")
    
    @validator('progress_percentage')
    def validate_progress(cls, v):
        return max(0.0, min(100.0, v))
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate session duration"""
        if self.completed_at:
            return self.completed_at - self.started_at
        return datetime.now() - self.started_at
    
    @property
    def success_rate(self) -> float:
        """Calculate scraping success rate"""
        total = self.listings_scraped + self.listings_failed
        if total == 0:
            return 0.0
        return (self.listings_scraped / total) * 100
    
    @property
    def requests_success_rate(self) -> float:
        """Calculate request success rate"""
        if self.requests_made == 0:
            return 0.0
        return (self.requests_successful / self.requests_made) * 100
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status in [ScrapingStatus.RUNNING, ScrapingStatus.PAUSED, ScrapingStatus.RETRYING]
    
    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        """Estimate time remaining based on current progress"""
        if self.progress_percentage <= 0:
            return None
        
        elapsed = self.duration
        if not elapsed:
            return None
        
        total_estimated = elapsed / (self.progress_percentage / 100)
        return total_estimated - elapsed
    
    def update_progress(self, percentage: float, current_phase: str = None):
        """Update session progress"""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        if current_phase:
            self.current_phase = current_phase
        self.last_activity = datetime.now()
        
        # Update estimated completion
        if self.progress_percentage > 0:
            time_remaining = self.estimated_time_remaining
            if time_remaining:
                self.estimated_completion = datetime.now() + time_remaining