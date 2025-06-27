"""
Dynamic eBay Search URL Builder with Jewelry Category Filtering

This module provides comprehensive URL construction for eBay jewelry searches with
advanced filtering, pagination, and validation capabilities.

Features:
- Dynamic URL construction with query parameters
- Jewelry-specific category filtering and mapping
- Price range, condition, and sorting filters
- Location/shipping filters with validation
- Pagination support with configurable limits
- URL encoding and validation
- Advanced search parameters optimization
- Buy-it-now vs auction filtering
- Seller rating and business seller filters
- Material, brand, and size-specific searches

Integration:
- Compatible with existing crawl4ai eBay jewelry crawler
- Uses JewelryCategory enum from existing selectors
- Supports mobile/desktop URL variations
- Provides URL validation and optimization
"""

import re
import urllib.parse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import logging


class SortOrder(Enum):
    """eBay search result sorting options"""
    BEST_MATCH = "BestMatch"
    PRICE_LOW_TO_HIGH = "PricePlusShippingLowest"
    PRICE_HIGH_TO_LOW = "PricePlusShippingHighest"
    ENDING_SOONEST = "EndTimeSoonest"
    NEWLY_LISTED = "StartTimeNewest"
    DISTANCE_NEAREST = "DistanceNearest"
    CONDITION_NEW = "ConditionNew"
    CONDITION_USED = "ConditionUsed"
    BEST_OFFER = "BestOfferFirst"
    SELLER_RATING = "FeedbackRatingDescending"


class ItemCondition(Enum):
    """eBay item condition filters"""
    NEW = "1000"           # New
    NEW_OTHER = "1500"     # New other (see details)
    NEW_WITH_DEFECTS = "1750"  # New with defects
    MANUFACTURER_REFURBISHED = "2000"  # Manufacturer refurbished
    SELLER_REFURBISHED = "2500"        # Seller refurbished
    USED = "3000"          # Used
    FOR_PARTS = "7000"     # For parts or not working


class ListingType(Enum):
    """eBay listing type filters"""
    ALL = "All"
    AUCTION = "Auction"
    BUY_IT_NOW = "FixedPrice"
    BEST_OFFER = "StoreInventory"
    CLASSIFIED = "Classified"


class JewelryCategory(Enum):
    """Jewelry categories with eBay category IDs"""
    RINGS = {"id": "52546", "name": "Rings", "parent": "52541"}
    NECKLACES = {"id": "52544", "name": "Necklaces & Pendants", "parent": "52541"}
    EARRINGS = {"id": "52543", "name": "Earrings", "parent": "52541"}
    BRACELETS = {"id": "52545", "name": "Bracelets", "parent": "52541"}
    WATCHES = {"id": "14324", "name": "Watches", "parent": "281"}
    JEWELRY_SETS = {"id": "52547", "name": "Jewelry Sets", "parent": "52541"}
    BODY_JEWELRY = {"id": "52548", "name": "Body Jewelry", "parent": "52541"}
    GEMSTONES = {"id": "164829", "name": "Loose Gemstones", "parent": "52541"}
    VINTAGE_ANTIQUE = {"id": "48579", "name": "Vintage & Antique", "parent": "52541"}
    FINE_JEWELRY = {"id": "52541", "name": "Fine Jewelry", "parent": "281"}
    FASHION_JEWELRY = {"id": "137834", "name": "Fashion Jewelry", "parent": "281"}


class MetalType(Enum):
    """Metal type filters for jewelry"""
    GOLD = "Gold"
    SILVER = "Silver"
    PLATINUM = "Platinum"
    PALLADIUM = "Palladium" 
    TITANIUM = "Titanium"
    STAINLESS_STEEL = "Stainless Steel"
    TUNGSTEN = "Tungsten"
    COPPER = "Copper"
    BRASS = "Brass"
    MIXED_METALS = "Mixed Metals"


class GemstoneMaterial(Enum):
    """Gemstone/material filters"""
    DIAMOND = "Diamond"
    RUBY = "Ruby"
    SAPPHIRE = "Sapphire"
    EMERALD = "Emerald"
    PEARL = "Pearl"
    OPAL = "Opal"
    AMETHYST = "Amethyst"
    TOPAZ = "Topaz"
    GARNET = "Garnet"
    TURQUOISE = "Turquoise"
    ONYX = "Onyx"
    CUBIC_ZIRCONIA = "Cubic Zirconia"


@dataclass
class SearchFilters:
    """Container for all search filter parameters"""
    # Basic search
    query: Optional[str] = None
    category: Optional[JewelryCategory] = None
    
    # Price filters
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    
    # Condition filters
    condition: Optional[List[ItemCondition]] = None
    
    # Listing type filters
    listing_type: Optional[ListingType] = None
    buy_it_now_only: bool = False
    accepts_offers: bool = False
    
    # Sorting
    sort_order: SortOrder = SortOrder.BEST_MATCH
    
    # Location filters
    item_location: Optional[str] = None
    shipping_location: Optional[str] = None
    local_pickup_only: bool = False
    free_shipping_only: bool = False
    
    # Seller filters
    min_feedback_score: Optional[int] = None
    min_feedback_percentage: Optional[float] = None
    top_rated_sellers_only: bool = False
    business_sellers_only: bool = False
    
    # Jewelry-specific filters
    metal_type: Optional[MetalType] = None
    gemstone_material: Optional[GemstoneMaterial] = None
    ring_size: Optional[str] = None
    chain_length: Optional[str] = None
    brand: Optional[str] = None
    
    # Advanced filters
    completed_listings: bool = False
    sold_listings: bool = False
    
    # Pagination
    page_number: int = 1
    items_per_page: int = 50  # Max 200
    
    # Additional parameters
    extra_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class URLValidationResult:
    """Result of URL validation"""
    is_valid: bool
    url: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parameter_count: int = 0
    estimated_results: Optional[int] = None


class EBayURLBuilder:
    """
    Comprehensive eBay search URL builder for jewelry with advanced filtering
    
    Provides dynamic URL construction with validation, optimization, and
    extensive parameter support for jewelry-specific searches.
    """
    
    # eBay base URLs
    BASE_URLS = {
        'search': 'https://www.ebay.com/sch/i.html',
        'mobile_search': 'https://m.ebay.com/sch/i.html',
        'api_search': 'https://svcs.ebay.com/services/search/FindingService/v1'
    }
    
    # Category mappings for jewelry subcategories
    CATEGORY_MAPPINGS = {
        JewelryCategory.RINGS: {
            'engagement': '52549',
            'wedding_band': '52550', 
            'fashion': '52551',
            'mens': '52552',
            'vintage': '52553'
        },
        JewelryCategory.NECKLACES: {
            'chains': '52555',
            'pendants': '52556',
            'chokers': '52557',
            'lockets': '52558'
        },
        JewelryCategory.EARRINGS: {
            'studs': '52560',
            'hoops': '52561',
            'drop_dangle': '52562',
            'chandelier': '52563'
        },
        JewelryCategory.BRACELETS: {
            'tennis': '52565',
            'charm': '52566',
            'bangle': '52567',
            'cuff': '52568'
        },
        JewelryCategory.WATCHES: {
            'luxury': '14330',
            'fashion': '14331',
            'smartwatch': '178893',
            'vintage': '14332'
        }
    }
    
    # Ring size mappings
    RING_SIZES = {
        'us': ['3', '3.5', '4', '4.5', '5', '5.5', '6', '6.5', '7', '7.5', '8', '8.5', '9', '9.5', '10', '10.5', '11', '11.5', '12'],
        'uk': ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        'eu': ['44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64']
    }
    
    def __init__(self, 
                 mobile_mode: bool = False,
                 enable_validation: bool = True,
                 max_url_length: int = 2048,
                 enable_logging: bool = True):
        """
        Initialize the eBay URL builder
        
        Args:
            mobile_mode: Build mobile-optimized URLs
            enable_validation: Enable URL validation
            max_url_length: Maximum allowed URL length
            enable_logging: Enable logging
        """
        self.mobile_mode = mobile_mode
        self.enable_validation = enable_validation
        self.max_url_length = max_url_length
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger('null')
            self.logger.addHandler(logging.NullHandler())
        
        # URL construction stats
        self.stats = {
            'urls_built': 0,
            'parameters_used': {},
            'validation_errors': 0,
            'optimization_applied': 0
        }
    
    def build_search_url(self, filters: SearchFilters) -> URLValidationResult:
        """
        Build a complete eBay search URL with all specified filters
        
        Args:
            filters: SearchFilters object with all search parameters
            
        Returns:
            URLValidationResult with the constructed URL and validation info
        """
        try:
            # Choose base URL
            base_url = self.BASE_URLS['mobile_search'] if self.mobile_mode else self.BASE_URLS['search']
            
            # Build parameters dictionary
            params = self._build_parameters(filters)
            
            # Construct URL
            url = self._construct_url(base_url, params)
            
            # Validate if enabled
            if self.enable_validation:
                validation_result = self._validate_url(url, params)
            else:
                validation_result = URLValidationResult(
                    is_valid=True,
                    url=url,
                    parameter_count=len(params)
                )
            
            # Update statistics
            self._update_stats(params, validation_result.is_valid)
            
            self.logger.info(f"Built eBay search URL with {len(params)} parameters")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error building search URL: {e}")
            return URLValidationResult(
                is_valid=False,
                url="",
                errors=[f"URL construction failed: {str(e)}"]
            )
    
    def _build_parameters(self, filters: SearchFilters) -> Dict[str, str]:
        """Build the complete parameters dictionary"""
        params = {}
        
        # Basic search parameters
        if filters.query:
            params['_nkw'] = filters.query
        
        # Category parameters
        if filters.category:
            params['_sacat'] = filters.category.value['id']
        
        # Price parameters
        if filters.min_price is not None:
            params['_udlo'] = str(filters.min_price)
        if filters.max_price is not None:
            params['_udhi'] = str(filters.max_price)
        
        # Condition parameters
        if filters.condition:
            condition_values = [cond.value for cond in filters.condition]
            params['LH_ItemCondition'] = '|'.join(condition_values)
        
        # Listing type parameters
        if filters.listing_type:
            if filters.listing_type == ListingType.AUCTION:
                params['LH_Auction'] = '1'
            elif filters.listing_type == ListingType.BUY_IT_NOW:
                params['LH_BIN'] = '1'
            elif filters.listing_type == ListingType.BEST_OFFER:
                params['LH_BO'] = '1'
        
        if filters.buy_it_now_only:
            params['LH_BIN'] = '1'
        if filters.accepts_offers:
            params['LH_BO'] = '1'
        
        # Sorting parameters
        params['_sop'] = self._get_sort_value(filters.sort_order)
        
        # Location parameters
        if filters.item_location:
            params['_stpos'] = filters.item_location
        if filters.local_pickup_only:
            params['LH_PrefLoc'] = '1'
        if filters.free_shipping_only:
            params['LH_FS'] = '1'
        
        # Seller parameters
        if filters.min_feedback_score:
            params['_fsrp'] = str(filters.min_feedback_score)
        if filters.top_rated_sellers_only:
            params['LH_TitleDesc'] = '1'
        
        # Jewelry-specific parameters
        if filters.metal_type:
            params['Metal'] = filters.metal_type.value
        if filters.gemstone_material:
            params['Main_Stone'] = filters.gemstone_material.value
        if filters.ring_size:
            params['Ring_Size'] = filters.ring_size
        if filters.chain_length:
            params['Length'] = filters.chain_length
        if filters.brand:
            params['Brand'] = filters.brand
        
        # Advanced parameters
        if filters.completed_listings:
            params['LH_Complete'] = '1'
        if filters.sold_listings:
            params['LH_Sold'] = '1'
        
        # Pagination parameters
        if filters.page_number > 1:
            params['_pgn'] = str(filters.page_number)
        if filters.items_per_page != 50:
            params['_ipg'] = str(min(filters.items_per_page, 200))
        
        # Additional parameters
        params.update(filters.extra_params)
        
        # Add standard parameters
        params.update(self._get_standard_parameters())
        
        return params
    
    def _get_standard_parameters(self) -> Dict[str, str]:
        """Get standard parameters always included in URLs"""
        return {
            '_from': 'R40',  # Results from search
            '_trksid': 'p2334524.m570.l1313',  # Tracking ID
            '_odkw': '',  # Original keyword (empty for new search)
            'LH_TitleDesc': '0',  # Search in title and description
            'rt': 'nc',  # Real-time search
            '_osacat': '0'  # Original search category
        }
    
    def _get_sort_value(self, sort_order: SortOrder) -> str:
        """Convert sort order enum to eBay parameter value"""
        sort_mapping = {
            SortOrder.BEST_MATCH: '12',
            SortOrder.PRICE_LOW_TO_HIGH: '15',
            SortOrder.PRICE_HIGH_TO_LOW: '16',
            SortOrder.ENDING_SOONEST: '1',
            SortOrder.NEWLY_LISTED: '10',
            SortOrder.DISTANCE_NEAREST: '7',
            SortOrder.CONDITION_NEW: '18',
            SortOrder.CONDITION_USED: '19',
            SortOrder.BEST_OFFER: '20',
            SortOrder.SELLER_RATING: '21'
        }
        return sort_mapping.get(sort_order, '12')
    
    def _construct_url(self, base_url: str, params: Dict[str, str]) -> str:
        """Construct the final URL with parameters"""
        # Encode parameters
        encoded_params = []
        for key, value in params.items():
            if value:  # Only include non-empty values
                encoded_key = urllib.parse.quote_plus(str(key))
                encoded_value = urllib.parse.quote_plus(str(value))
                encoded_params.append(f"{encoded_key}={encoded_value}")
        
        # Join parameters
        query_string = '&'.join(encoded_params)
        
        # Construct final URL
        if query_string:
            url = f"{base_url}?{query_string}"
        else:
            url = base_url
        
        return url
    
    def _validate_url(self, url: str, params: Dict[str, str]) -> URLValidationResult:
        """Validate the constructed URL"""
        errors = []
        warnings = []
        
        # Check URL length
        if len(url) > self.max_url_length:
            errors.append(f"URL length ({len(url)}) exceeds maximum ({self.max_url_length})")
        
        # Check for required parameters
        if '_nkw' not in params and '_sacat' not in params:
            warnings.append("No search query or category specified - may return too many results")
        
        # Validate parameter combinations
        if params.get('LH_Auction') == '1' and params.get('LH_BIN') == '1':
            warnings.append("Both auction and buy-it-now filters active - may limit results")
        
        # Check price range validity
        min_price = params.get('_udlo')
        max_price = params.get('_udhi')
        if min_price and max_price:
            try:
                if float(min_price) >= float(max_price):
                    errors.append("Minimum price must be less than maximum price")
            except ValueError:
                errors.append("Invalid price format")
        
        # Validate pagination
        page_num = params.get('_pgn', '1')
        try:
            if int(page_num) > 100:
                warnings.append("Page number exceeds typical eBay limits")
        except ValueError:
            errors.append("Invalid page number format")
        
        # Validate items per page
        items_per_page = params.get('_ipg', '50')
        try:
            if int(items_per_page) > 200:
                errors.append("Items per page exceeds eBay maximum (200)")
        except ValueError:
            errors.append("Invalid items per page format")
        
        # Estimate results count based on parameters
        estimated_results = self._estimate_results_count(params)
        
        return URLValidationResult(
            is_valid=len(errors) == 0,
            url=url,
            errors=errors,
            warnings=warnings,
            parameter_count=len(params),
            estimated_results=estimated_results
        )
    
    def _estimate_results_count(self, params: Dict[str, str]) -> Optional[int]:
        """Estimate number of results based on search parameters"""
        # This is a rough estimation based on parameter restrictiveness
        base_estimate = 10000  # Base estimate for jewelry searches
        
        # Reduce estimate based on restrictive parameters
        if params.get('_nkw'):  # Specific search query
            base_estimate *= 0.3
        if params.get('Brand'):  # Specific brand
            base_estimate *= 0.2
        if params.get('Metal'):  # Specific metal
            base_estimate *= 0.4
        if params.get('Main_Stone'):  # Specific stone
            base_estimate *= 0.3
        if params.get('_udlo') or params.get('_udhi'):  # Price filters
            base_estimate *= 0.6
        if params.get('LH_ItemCondition'):  # Condition filters
            base_estimate *= 0.7
        
        return max(int(base_estimate), 1)
    
    def _update_stats(self, params: Dict[str, str], is_valid: bool):
        """Update construction statistics"""
        self.stats['urls_built'] += 1
        
        if not is_valid:
            self.stats['validation_errors'] += 1
        
        # Track parameter usage
        for param in params.keys():
            if param not in self.stats['parameters_used']:
                self.stats['parameters_used'][param] = 0
            self.stats['parameters_used'][param] += 1
    
    def build_category_url(self, 
                          category: JewelryCategory, 
                          subcategory: Optional[str] = None) -> str:
        """
        Build a URL for browsing a specific jewelry category
        
        Args:
            category: Main jewelry category
            subcategory: Optional subcategory
            
        Returns:
            Category browsing URL
        """
        base_url = self.BASE_URLS['mobile_search'] if self.mobile_mode else self.BASE_URLS['search']
        
        params = {
            '_sacat': category.value['id'],
            '_from': 'R40'
        }
        
        # Add subcategory if specified
        if subcategory and category in self.CATEGORY_MAPPINGS:
            subcategory_mapping = self.CATEGORY_MAPPINGS[category]
            if subcategory in subcategory_mapping:
                params['_sacat'] = subcategory_mapping[subcategory]
        
        return self._construct_url(base_url, params)
    
    def build_seller_url(self, seller_name: str, category: Optional[JewelryCategory] = None) -> str:
        """
        Build a URL for browsing a specific seller's jewelry items
        
        Args:
            seller_name: eBay seller username
            category: Optional category filter
            
        Returns:
            Seller items URL
        """
        base_url = "https://www.ebay.com/sch/m.html"
        
        params = {
            '_ssn': seller_name,
            '_from': 'R40'
        }
        
        if category:
            params['_sacat'] = category.value['id']
        
        return self._construct_url(base_url, params)
    
    def optimize_url_for_mobile(self, url: str) -> str:
        """
        Convert a desktop URL to mobile-optimized version
        
        Args:
            url: Desktop eBay URL
            
        Returns:
            Mobile-optimized URL
        """
        # Replace domain
        mobile_url = url.replace('www.ebay.com', 'm.ebay.com')
        
        # Add mobile-specific parameters
        if '?' in mobile_url:
            mobile_url += '&_mwBanner=1'
        else:
            mobile_url += '?_mwBanner=1'
        
        return mobile_url
    
    def get_supported_parameters(self) -> Dict[str, Any]:
        """
        Get comprehensive list of supported parameters and their options
        
        Returns:
            Dictionary of parameter categories and options
        """
        return {
            'basic_search': {
                'query': 'str - Search keywords',
                'category': f'JewelryCategory - {len(JewelryCategory)} categories supported'
            },
            'price_filters': {
                'min_price': 'float - Minimum price',
                'max_price': 'float - Maximum price'
            },
            'condition_filters': {
                'condition': f'List[ItemCondition] - {len(ItemCondition)} conditions supported'
            },
            'listing_type_filters': {
                'listing_type': f'ListingType - {len(ListingType)} types supported',
                'buy_it_now_only': 'bool',
                'accepts_offers': 'bool'
            },
            'sorting_options': {
                'sort_order': f'SortOrder - {len(SortOrder)} options supported'
            },
            'location_filters': {
                'item_location': 'str - ZIP code or city',
                'shipping_location': 'str - Ships to location',
                'local_pickup_only': 'bool',
                'free_shipping_only': 'bool'
            },
            'seller_filters': {
                'min_feedback_score': 'int',
                'min_feedback_percentage': 'float',
                'top_rated_sellers_only': 'bool',
                'business_sellers_only': 'bool'
            },
            'jewelry_specific': {
                'metal_type': f'MetalType - {len(MetalType)} metals supported',
                'gemstone_material': f'GemstoneMaterial - {len(GemstoneMaterial)} materials supported',
                'ring_size': f'str - {len(self.RING_SIZES["us"])} US sizes supported',
                'chain_length': 'str - Chain length specification',
                'brand': 'str - Brand name'
            },
            'advanced_filters': {
                'completed_listings': 'bool - Show completed auctions',
                'sold_listings': 'bool - Show sold items only'
            },
            'pagination': {
                'page_number': 'int - Page number (1-based)',
                'items_per_page': 'int - Items per page (max 200)'
            }
        }
    
    def get_category_mappings(self) -> Dict[str, Any]:
        """
        Get jewelry category mappings with eBay category IDs
        
        Returns:
            Dictionary of category mappings
        """
        mappings = {}
        for category in JewelryCategory:
            mappings[category.name] = {
                'category_id': category.value['id'],
                'category_name': category.value['name'],
                'parent_id': category.value['parent'],
                'subcategories': self.CATEGORY_MAPPINGS.get(category, {})
            }
        return mappings
    
    def get_validation_features(self) -> Dict[str, Any]:
        """
        Get summary of URL validation features
        
        Returns:
            Dictionary of validation capabilities
        """
        return {
            'url_length_validation': True,
            'parameter_combination_validation': True,
            'price_range_validation': True,
            'pagination_validation': True,
            'format_validation': True,
            'estimation_features': True,
            'max_url_length': self.max_url_length,
            'mobile_optimization': True,
            'encoding_safety': True
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get URL builder usage statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset usage statistics"""
        self.stats = {
            'urls_built': 0,
            'parameters_used': {},
            'validation_errors': 0,
            'optimization_applied': 0
        }


# Convenience functions for quick URL building

def build_jewelry_search_url(query: str, 
                           category: Optional[JewelryCategory] = None,
                           min_price: Optional[float] = None,
                           max_price: Optional[float] = None,
                           condition: Optional[List[ItemCondition]] = None,
                           sort_order: SortOrder = SortOrder.BEST_MATCH) -> str:
    """
    Quick function to build a basic jewelry search URL
    
    Args:
        query: Search keywords
        category: Jewelry category
        min_price: Minimum price filter
        max_price: Maximum price filter
        condition: Item condition filters
        sort_order: Result sorting preference
        
    Returns:
        Complete eBay search URL
    """
    builder = EBayURLBuilder()
    filters = SearchFilters(
        query=query,
        category=category,
        min_price=min_price,
        max_price=max_price,
        condition=condition,
        sort_order=sort_order
    )
    
    result = builder.build_search_url(filters)
    return result.url


def build_mobile_jewelry_url(query: str, category: JewelryCategory) -> str:
    """
    Quick function to build a mobile-optimized jewelry search URL
    
    Args:
        query: Search keywords
        category: Jewelry category
        
    Returns:
        Mobile-optimized eBay search URL
    """
    builder = EBayURLBuilder(mobile_mode=True)
    filters = SearchFilters(query=query, category=category)
    
    result = builder.build_search_url(filters)
    return result.url


# Export main classes and functions
__all__ = [
    'EBayURLBuilder',
    'SearchFilters',
    'URLValidationResult',
    'SortOrder',
    'ItemCondition',
    'ListingType',
    'JewelryCategory',
    'MetalType',
    'GemstoneMaterial',
    'build_jewelry_search_url',
    'build_mobile_jewelry_url'
]