"""
Jewelry Scraper Database Models
SQLite schema and Pydantic models for jewelry listings, images, specifications, and scraping sessions.
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from urllib.parse import urlparse


class JewelryCategory(str, Enum):
    """Jewelry category classification"""
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
    OTHER = "other"


class JewelryMaterial(str, Enum):
    """Jewelry material types"""
    GOLD = "gold"
    SILVER = "silver"
    PLATINUM = "platinum"
    TITANIUM = "titanium"
    STAINLESS_STEEL = "stainless_steel"
    COPPER = "copper"
    BRASS = "brass"
    LEATHER = "leather"
    FABRIC = "fabric"
    PLASTIC = "plastic"
    CERAMIC = "ceramic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ListingStatus(str, Enum):
    """eBay listing status"""
    ACTIVE = "active"
    SOLD = "sold"
    ENDED = "ended"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


class ScrapingStatus(str, Enum):
    """Scraping session status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ImageType(str, Enum):
    """Image classification types"""
    MAIN = "main"
    GALLERY = "gallery"
    DETAIL = "detail"
    CERTIFICATE = "certificate"
    PACKAGING = "packaging"
    COMPARISON = "comparison"


class JewelryListing(BaseModel):
    """
    Comprehensive jewelry listing data model with Pydantic validation
    Represents a single eBay jewelry listing with all extracted information
    """
    
    # Required Basic Fields
    id: str = Field(..., description="Unique listing identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Listing title")
    price: float = Field(..., gt=0, description="Current price (must be positive)")
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$", description="ISO currency code")
    condition: str = Field(..., min_length=1, description="Item condition")
    
    # eBay Specific Required Fields
    seller_id: str = Field(..., min_length=1, description="eBay seller username/ID")
    listing_url: str = Field(..., description="Full eBay listing URL")
    end_time: Optional[datetime] = Field(None, description="Listing end time")
    shipping_cost: Optional[float] = Field(None, ge=0, description="Shipping cost (non-negative)")
    
    # Jewelry Specific Required Fields
    category: JewelryCategory = Field(..., description="Jewelry category")
    material: JewelryMaterial = Field(..., description="Primary jewelry material")
    gemstone: Optional[str] = Field(None, description="Primary gemstone type")
    size: Optional[str] = Field(None, description="Size information")
    weight: Optional[str] = Field(None, description="Weight information")
    brand: Optional[str] = Field(None, description="Brand name")
    
    # Images Required Fields
    image_urls: List[str] = Field(default_factory=list, description="List of all image URLs")
    main_image_path: Optional[str] = Field(None, description="Path to main/primary image")
    
    # Metadata Required Fields
    scraped_at: datetime = Field(default_factory=datetime.now, description="Timestamp when scraped")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data quality score (0-1)")
    
    # Additional Optional Fields
    listing_id: Optional[str] = Field(None, description="Original eBay listing ID")
    original_price: Optional[float] = Field(None, gt=0, description="Original/retail price")
    availability: Optional[str] = Field(None, description="Availability status")
    
    # Seller information
    seller_rating: Optional[float] = Field(None, ge=0, le=100, description="Seller feedback score")
    seller_feedback_count: Optional[int] = Field(None, ge=0, description="Number of feedback items")
    
    # Categorization
    subcategory: Optional[str] = Field(None, description="Specific subcategory")
    
    # Product specifications
    materials: List[str] = Field(default_factory=list, description="All materials mentioned")
    dimensions: Optional[str] = Field(None, description="Dimensions")
    
    # Additional gemstone information
    stone_color: Optional[str] = Field(None, description="Primary stone color")
    stone_clarity: Optional[str] = Field(None, description="Stone clarity")
    stone_cut: Optional[str] = Field(None, description="Stone cut")
    stone_carat: Optional[str] = Field(None, description="Stone carat weight")
    
    # Additional stones
    accent_stones: List[str] = Field(default_factory=list, description="Accent/secondary stones")
    
    # Description and features
    description: Optional[str] = Field(None, description="Full item description")
    features: List[str] = Field(default_factory=list, description="Key features")
    tags: List[str] = Field(default_factory=list, description="Generated tags")
    
    # eBay specific
    item_number: Optional[str] = Field(None, description="eBay item number")
    listing_type: Optional[str] = Field(None, description="Auction, Buy It Now, etc.")
    listing_status: ListingStatus = Field(default=ListingStatus.UNKNOWN, description="Current listing status")
    watchers: Optional[int] = Field(None, description="Number of watchers")
    views: Optional[int] = Field(None, description="Number of views")
    bids: Optional[int] = Field(None, description="Number of bids")
    time_left: Optional[str] = Field(None, description="Time remaining")
    
    # Shipping and location  
    ships_from: Optional[str] = Field(None, description="Shipping location")
    ships_to: Optional[str] = Field(None, description="Shipping destinations")
    
    # Quality metrics
    image_count: int = Field(default=0, ge=0, description="Number of images")
    description_length: int = Field(default=0, ge=0, description="Description character count")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    listing_date: Optional[datetime] = Field(None, description="When listing was posted")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw scraped data")
    
    # Validation
    is_validated: bool = Field(default=False, description="Data validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    
    # === COMPREHENSIVE PYDANTIC VALIDATORS ===
    
    @validator('listing_url')
    def validate_listing_url(cls, v):
        """Validate eBay listing URL format"""
        if not v:
            raise ValueError('Listing URL is required')
        
        # Check if it's a valid URL
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Invalid URL format')
        except Exception:
            raise ValueError('Invalid URL format')
        
        # Check if it's an eBay URL
        if 'ebay.' not in v.lower():
            raise ValueError('URL must be from eBay domain')
        
        return v
    
    @validator('image_urls')
    def validate_image_urls(cls, v):
        """Validate all image URLs are properly formatted"""
        if not v:
            return v
        
        valid_urls = []
        for url in v:
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    valid_urls.append(url)
            except Exception:
                continue  # Skip invalid URLs but don't fail validation
        
        return valid_urls
    
    @validator('main_image_path')
    def validate_main_image_path(cls, v):
        """Validate main image path format"""
        if v and not isinstance(v, str):
            raise ValueError('Main image path must be a string')
        return v
    
    @validator('seller_id')
    def validate_seller_id(cls, v):
        """Validate seller ID format"""
        if not v or not v.strip():
            raise ValueError('Seller ID cannot be empty')
        
        # Remove common eBay seller ID prefixes/suffixes
        cleaned = v.strip().replace('seller_', '').replace('_seller', '')
        if len(cleaned) < 3:
            raise ValueError('Seller ID must be at least 3 characters')
        
        return v.strip()
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code format"""
        if not re.match(r'^[A-Z]{3}$', v):
            raise ValueError('Currency must be a 3-letter ISO code (e.g., USD, EUR)')
        return v.upper()
    
    @validator('title')
    def validate_title(cls, v):
        """Validate and clean title"""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        
        # Clean common eBay title artifacts
        cleaned = v.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
        
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + '...'
        
        return cleaned
    
    @validator('condition')
    def validate_condition(cls, v):
        """Validate and standardize condition"""
        if not v or not v.strip():
            raise ValueError('Condition cannot be empty')
        
        # Standardize common condition values
        condition_map = {
            'new': 'New',
            'new with tags': 'New with tags',
            'new without tags': 'New without tags',
            'used': 'Used',
            'very good': 'Very Good',
            'good': 'Good',
            'acceptable': 'Acceptable',
            'for parts': 'For parts or not working'
        }
        
        cleaned = v.strip().lower()
        return condition_map.get(cleaned, v.strip())
    
    def calculate_data_quality_score(self) -> float:
        """
        Enhanced data quality scoring method
        Returns score from 0.0 to 1.0 based on field completeness and quality
        """
        score = 0.0
        
        # Required fields (40% weight) - must be present
        required_fields = ['id', 'title', 'price', 'currency', 'condition', 'seller_id', 'listing_url', 'category', 'material']
        required_filled = sum(1 for field in required_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (required_filled / len(required_fields)) * 0.4
        
        # Important fields (35% weight) - significantly improve data value
        important_fields = ['brand', 'gemstone', 'size', 'weight', 'description', 'image_urls', 'shipping_cost']
        important_filled = sum(1 for field in important_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (important_filled / len(important_fields)) * 0.35
        
        # Enhancement fields (15% weight) - add extra value
        enhancement_fields = ['stone_color', 'stone_clarity', 'stone_cut', 'features', 'dimensions']
        enhancement_filled = sum(1 for field in enhancement_fields if self._is_field_valuable(getattr(self, field, None)))
        score += (enhancement_filled / len(enhancement_fields)) * 0.15
        
        # Quality bonuses (10% weight)
        quality_score = 0.0
        
        # Bonus for image count
        if self.image_count > 0:
            quality_score += min(self.image_count / 10, 0.3)  # Max 30% of quality score for images
        
        # Bonus for description length
        if self.description_length > 50:
            quality_score += min(self.description_length / 1000, 0.3)  # Max 30% for description
        
        # Bonus for having main image path
        if self.main_image_path:
            quality_score += 0.2
        
        # Bonus for end time (shows active listing)
        if self.end_time:
            quality_score += 0.2
        
        score += min(quality_score, 1.0) * 0.1  # Cap quality bonuses at 10%
        
        return round(min(score, 1.0), 3)  # Ensure max 1.0 and 3 decimal precision
    
    def _is_field_valuable(self, value) -> bool:
        """Check if a field has valuable content"""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict)):
            return len(value) > 0
        if isinstance(value, (int, float)):
            return value > 0
        return bool(value)
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert to dictionary with optional metadata inclusion"""
        data = self.dict()
        
        if not include_metadata:
            # Remove internal fields for cleaner output
            data.pop('raw_data', None)
            data.pop('validation_errors', None)
            data.pop('is_validated', None)
        
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data
    
    def to_json(self, include_metadata: bool = False) -> str:
        """Convert to JSON string with proper datetime serialization"""
        return json.dumps(self.to_dict(include_metadata), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JewelryListing':
        """Create instance from dictionary with type conversion"""
        # Convert ISO datetime strings back to datetime objects
        datetime_fields = ['scraped_at', 'created_at', 'updated_at', 'listing_date', 'end_time']
        
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except ValueError:
                    data[field] = None
        
        return cls(**data)
    
    def update_quality_score(self):
        """Update the data quality score and validation status"""
        self.data_quality_score = self.calculate_data_quality_score()
        self.updated_at = datetime.now()
        
        # Auto-validate if quality score is high enough
        if self.data_quality_score >= 0.7:
            self.is_validated = True
            self.validation_errors = []
    
    def validate_for_database(self) -> bool:
        """Validate the model is ready for database insertion"""
        errors = []
        
        # Check required fields
        if not self.id:
            errors.append("ID is required")
        if not self.title:
            errors.append("Title is required")
        if not self.price or self.price <= 0:
            errors.append("Valid price is required")
        if not self.seller_id:
            errors.append("Seller ID is required")
        if not self.listing_url:
            errors.append("Listing URL is required")
        
        self.validation_errors = errors
        self.is_validated = len(errors) == 0
        
        return self.is_validated
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the jewelry listing"""
        return {
            'id': self.id,
            'title': self.title[:100] + '...' if len(self.title) > 100 else self.title,
            'price': f"{self.currency} {self.price:.2f}",
            'category': self.category.value,
            'material': self.material.value,
            'brand': self.brand or 'Unknown',
            'condition': self.condition,
            'seller': self.seller_id,
            'images': len(self.image_urls),
            'quality_score': f"{self.data_quality_score:.1%}",
            'scraped': self.scraped_at.strftime('%Y-%m-%d %H:%M'),
            'validated': self.is_validated
        }


class JewelryImage(BaseModel):
    """
    Image metadata and storage information
    """
    
    # Primary identifiers
    image_id: str = Field(..., description="Unique image identifier")
    listing_id: str = Field(..., description="Associated listing ID")
    
    # Image source and storage
    original_url: str = Field(..., description="Original image URL from eBay")
    local_path: Optional[str] = Field(None, description="Local file path")
    filename: Optional[str] = Field(None, description="Generated filename")
    
    # Image classification
    image_type: ImageType = Field(default=ImageType.GALLERY, description="Image type/purpose")
    sequence_order: int = Field(default=0, description="Order in listing gallery")
    
    # Image properties
    file_size: Optional[int] = Field(None, description="File size in bytes")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    format: Optional[str] = Field(None, description="Image format (jpg, png, etc.)")
    
    # Quality and processing
    is_processed: bool = Field(default=False, description="Image processing status")
    is_optimized: bool = Field(default=False, description="Image optimization status")
    quality_score: Optional[float] = Field(None, description="Image quality assessment (0-1)")
    
    # Content analysis
    contains_text: bool = Field(default=False, description="Whether image contains text")
    is_duplicate: bool = Field(default=False, description="Duplicate detection flag")
    similarity_hash: Optional[str] = Field(None, description="Perceptual hash for similarity")
    
    # Alt text and descriptions
    alt_text: Optional[str] = Field(None, description="Alt text from HTML")
    generated_description: Optional[str] = Field(None, description="AI-generated description")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation time")
    downloaded_at: Optional[datetime] = Field(None, description="Download timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        if v is not None:
            return max(0.0, min(1.0, v))
        return v


class JewelrySpecification(BaseModel):
    """
    Detailed specifications and attributes for jewelry items
    """
    
    # Primary identifiers
    spec_id: str = Field(..., description="Unique specification ID")
    listing_id: str = Field(..., description="Associated listing ID")
    
    # Specification details
    attribute_name: str = Field(..., description="Specification attribute name")
    attribute_value: str = Field(..., description="Specification value")
    attribute_category: Optional[str] = Field(None, description="Category of specification")
    
    # Data source and confidence
    source_section: Optional[str] = Field(None, description="Where this was extracted from")
    confidence_score: float = Field(default=0.0, description="Extraction confidence (0-1)")
    
    # Standardization
    standardized_name: Optional[str] = Field(None, description="Standardized attribute name")
    standardized_value: Optional[str] = Field(None, description="Standardized value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    
    # Validation
    is_verified: bool = Field(default=False, description="Manual verification status")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation time")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        return max(0.0, min(1.0, v))


class ScrapingSession(BaseModel):
    """
    Scraping session tracking and statistics
    """
    
    # Primary identifiers
    session_id: str = Field(..., description="Unique session identifier")
    session_name: Optional[str] = Field(None, description="Human-readable session name")
    
    # Session configuration
    search_query: Optional[str] = Field(None, description="Search query used")
    search_filters: Dict[str, Any] = Field(default_factory=dict, description="Applied search filters")
    max_pages: Optional[int] = Field(None, description="Maximum pages to scrape")
    max_listings: Optional[int] = Field(None, description="Maximum listings to scrape")
    
    # Session status
    status: ScrapingStatus = Field(default=ScrapingStatus.RUNNING, description="Current session status")
    progress_percentage: float = Field(default=0.0, description="Progress percentage (0-100)")
    
    # Statistics
    listings_found: int = Field(default=0, description="Total listings discovered")
    listings_scraped: int = Field(default=0, description="Successfully scraped listings")
    listings_failed: int = Field(default=0, description="Failed scraping attempts")
    images_downloaded: int = Field(default=0, description="Images successfully downloaded")
    images_failed: int = Field(default=0, description="Failed image downloads")
    
    # Performance metrics
    pages_processed: int = Field(default=0, description="Search result pages processed")
    requests_made: int = Field(default=0, description="Total HTTP requests made")
    data_volume_mb: float = Field(default=0.0, description="Total data downloaded (MB)")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now, description="Session start time")
    completed_at: Optional[datetime] = Field(None, description="Session completion time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    
    # Error handling
    error_count: int = Field(default=0, description="Total errors encountered")
    last_error: Optional[str] = Field(None, description="Most recent error message")
    retry_count: int = Field(default=0, description="Number of retries performed")
    
    # Configuration and metadata
    user_agent: Optional[str] = Field(None, description="User agent used")
    proxy_used: Optional[str] = Field(None, description="Proxy configuration")
    rate_limit_delay: float = Field(default=1.0, description="Rate limiting delay (seconds)")
    
    # Output configuration
    export_formats: List[str] = Field(default_factory=list, description="Requested export formats")
    output_directory: Optional[str] = Field(None, description="Output directory path")
    
    # Session metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
    
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
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status in [ScrapingStatus.RUNNING, ScrapingStatus.PAUSED]


# Database table creation SQL schemas
JEWELRY_SCHEMA_SQL = {
    "listings": """
        CREATE TABLE IF NOT EXISTS jewelry_listings (
            listing_id TEXT PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            price REAL,
            original_price REAL,
            currency TEXT DEFAULT 'USD',
            condition TEXT,
            availability TEXT,
            seller_name TEXT,
            seller_rating REAL,
            seller_feedback_count INTEGER,
            category TEXT NOT NULL,
            subcategory TEXT,
            brand TEXT,
            material TEXT NOT NULL,
            materials TEXT, -- JSON array
            size TEXT,
            weight TEXT,
            dimensions TEXT,
            main_stone TEXT,
            stone_color TEXT,
            stone_clarity TEXT,
            stone_cut TEXT,
            stone_carat TEXT,
            accent_stones TEXT, -- JSON array
            description TEXT,
            features TEXT, -- JSON array
            tags TEXT, -- JSON array
            item_number TEXT,
            listing_type TEXT,
            listing_status TEXT DEFAULT 'unknown',
            watchers INTEGER,
            views INTEGER,
            bids INTEGER,
            time_left TEXT,
            shipping_cost REAL,
            ships_from TEXT,
            ships_to TEXT,
            image_count INTEGER DEFAULT 0,
            description_length INTEGER DEFAULT 0,
            data_completeness_score REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            listing_date TIMESTAMP,
            metadata TEXT, -- JSON object
            raw_data TEXT, -- JSON object
            is_validated BOOLEAN DEFAULT FALSE,
            validation_errors TEXT -- JSON array
        )
    """,
    
    "images": """
        CREATE TABLE IF NOT EXISTS jewelry_images (
            image_id TEXT PRIMARY KEY,
            listing_id TEXT NOT NULL,
            original_url TEXT NOT NULL,
            local_path TEXT,
            filename TEXT,
            image_type TEXT DEFAULT 'gallery',
            sequence_order INTEGER DEFAULT 0,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            format TEXT,
            is_processed BOOLEAN DEFAULT FALSE,
            is_optimized BOOLEAN DEFAULT FALSE,
            quality_score REAL,
            contains_text BOOLEAN DEFAULT FALSE,
            is_duplicate BOOLEAN DEFAULT FALSE,
            similarity_hash TEXT,
            alt_text TEXT,
            generated_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            downloaded_at TIMESTAMP,
            processed_at TIMESTAMP,
            metadata TEXT, -- JSON object
            FOREIGN KEY (listing_id) REFERENCES jewelry_listings (listing_id) ON DELETE CASCADE
        )
    """,
    
    "specifications": """
        CREATE TABLE IF NOT EXISTS jewelry_specifications (
            spec_id TEXT PRIMARY KEY,
            listing_id TEXT NOT NULL,
            attribute_name TEXT NOT NULL,
            attribute_value TEXT NOT NULL,
            attribute_category TEXT,
            source_section TEXT,
            confidence_score REAL DEFAULT 0.0,
            standardized_name TEXT,
            standardized_value TEXT,
            unit TEXT,
            is_verified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT, -- JSON object
            FOREIGN KEY (listing_id) REFERENCES jewelry_listings (listing_id) ON DELETE CASCADE
        )
    """,
    
    "scraping_sessions": """
        CREATE TABLE IF NOT EXISTS scraping_sessions (
            session_id TEXT PRIMARY KEY,
            session_name TEXT,
            search_query TEXT,
            search_filters TEXT, -- JSON object
            max_pages INTEGER,
            max_listings INTEGER,
            status TEXT DEFAULT 'running',
            progress_percentage REAL DEFAULT 0.0,
            listings_found INTEGER DEFAULT 0,
            listings_scraped INTEGER DEFAULT 0,
            listings_failed INTEGER DEFAULT 0,
            images_downloaded INTEGER DEFAULT 0,
            images_failed INTEGER DEFAULT 0,
            pages_processed INTEGER DEFAULT 0,
            requests_made INTEGER DEFAULT 0,
            data_volume_mb REAL DEFAULT 0.0,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_count INTEGER DEFAULT 0,
            last_error TEXT,
            retry_count INTEGER DEFAULT 0,
            user_agent TEXT,
            proxy_used TEXT,
            rate_limit_delay REAL DEFAULT 1.0,
            export_formats TEXT, -- JSON array
            output_directory TEXT,
            metadata TEXT -- JSON object
        )
    """
}

# Index creation SQL for performance optimization
JEWELRY_INDEXES_SQL = [
    # Listings table indexes
    "CREATE INDEX IF NOT EXISTS idx_listings_category ON jewelry_listings(category)",
    "CREATE INDEX IF NOT EXISTS idx_listings_material ON jewelry_listings(material)",
    "CREATE INDEX IF NOT EXISTS idx_listings_price ON jewelry_listings(price)",
    "CREATE INDEX IF NOT EXISTS idx_listings_brand ON jewelry_listings(brand)",
    "CREATE INDEX IF NOT EXISTS idx_listings_seller ON jewelry_listings(seller_name)",
    "CREATE INDEX IF NOT EXISTS idx_listings_scraped_at ON jewelry_listings(scraped_at)",
    "CREATE INDEX IF NOT EXISTS idx_listings_status ON jewelry_listings(listing_status)",
    "CREATE INDEX IF NOT EXISTS idx_listings_completeness ON jewelry_listings(data_completeness_score)",
    
    # Images table indexes
    "CREATE INDEX IF NOT EXISTS idx_images_listing_id ON jewelry_images(listing_id)",
    "CREATE INDEX IF NOT EXISTS idx_images_type ON jewelry_images(image_type)",
    "CREATE INDEX IF NOT EXISTS idx_images_processed ON jewelry_images(is_processed)",
    "CREATE INDEX IF NOT EXISTS idx_images_duplicate ON jewelry_images(is_duplicate)",
    "CREATE INDEX IF NOT EXISTS idx_images_hash ON jewelry_images(similarity_hash)",
    
    # Specifications table indexes
    "CREATE INDEX IF NOT EXISTS idx_specs_listing_id ON jewelry_specifications(listing_id)",
    "CREATE INDEX IF NOT EXISTS idx_specs_name ON jewelry_specifications(attribute_name)",
    "CREATE INDEX IF NOT EXISTS idx_specs_category ON jewelry_specifications(attribute_category)",
    "CREATE INDEX IF NOT EXISTS idx_specs_confidence ON jewelry_specifications(confidence_score)",
    
    # Scraping sessions table indexes
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON scraping_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON scraping_sessions(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_query ON scraping_sessions(search_query)"
]