"""
Shared Types for eBay Jewelry Scraper

Common data types and enums used across the scraper modules to avoid circular imports.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import jewelry models using relative imports
from .jewelry_models import JewelryListing


class ScrapingMode(Enum):
    """Scraping operation modes"""
    SEARCH_RESULTS = "search_results"
    INDIVIDUAL_LISTING = "individual_listing"
    CATEGORY_BROWSE = "category_browse"
    SELLER_LISTINGS = "seller_listings"


class AntiDetectionLevel(Enum):
    """Anti-detection strictness levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    STEALTH = "stealth"


@dataclass
class ScrapingResult:
    """Result container for scraping operations"""

    success: bool
    data: Optional[Union[JewelryListing, List[JewelryListing]]] = None
    error: Optional[str] = None
    response_time: float = 0.0
    retry_count: int = 0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'success': self.success,
            'error': self.error,
            'response_time': self.response_time,
            'retry_count': self.retry_count,
            'quality_score': self.quality_score,
            'metadata': self.metadata
        }

        if self.data:
            if isinstance(self.data, list):
                result['data'] = [item.to_dict() for item in self.data]
            else:
                result['data'] = self.data.to_dict()

        return result
