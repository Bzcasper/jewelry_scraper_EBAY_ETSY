"""
Comprehensive CSS Selectors Dictionary for eBay Jewelry Data Extraction

This module provides a robust, performance-optimized selector management system
for extracting jewelry-specific data from eBay listings with high reliability.

Features:
- Primary + fallback selectors for maximum reliability
- Success rate tracking and optimization
- Jewelry category-specific variations
- Mobile vs desktop responsive selectors
- Performance-optimized selector patterns
"""

import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict


class SelectorType(Enum):
    """Types of selectors for different extraction purposes"""
    SEARCH_RESULTS = "search_results"
    PRODUCT_DETAILS = "product_details"
    JEWELRY_SPECIFIC = "jewelry_specific"
    IMAGES = "images"
    SHIPPING = "shipping"
    SELLER = "seller"
    MOBILE_SPECIFIC = "mobile_specific"


class DeviceType(Enum):
    """Device types for responsive selector targeting"""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    ALL = "all"


class JewelryCategory(Enum):
    """Jewelry categories for specialized selectors"""
    RINGS = "rings"
    NECKLACES = "necklaces"
    EARRINGS = "earrings"
    BRACELETS = "bracelets"
    WATCHES = "watches"
    GEMSTONES = "gemstones"
    OTHER = "other"


@dataclass
class SelectorInfo:
    """Container for selector information and metadata"""
    primary: str
    fallbacks: List[str]
    description: str
    device_type: DeviceType = DeviceType.ALL
    category_specific: Optional[JewelryCategory] = None
    performance_weight: int = 1  # Lower is better for performance
    last_tested: Optional[datetime] = None
    success_rate: float = 0.0
    total_attempts: int = 0
    successful_extractions: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_tested:
            data['last_tested'] = self.last_tested.isoformat()
        if self.device_type:
            data['device_type'] = self.device_type.value
        if self.category_specific:
            data['category_specific'] = self.category_specific.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SelectorInfo':
        """Create from dictionary"""
        data = data.copy()
        if data.get('last_tested'):
            data['last_tested'] = datetime.fromisoformat(data['last_tested'])
        if data.get('device_type'):
            data['device_type'] = DeviceType(data['device_type'])
        if data.get('category_specific'):
            data['category_specific'] = JewelryCategory(data['category_specific'])
        return cls(**data)


class SelectorManager:
    """
    Comprehensive CSS selector management system for eBay jewelry extraction
    
    Provides intelligent selector fallback, success tracking, and optimization
    for reliable data extraction across different eBay page layouts and devices.
    """
    
    def __init__(self, 
                 cache_file: Optional[str] = "./selectors_cache.json",
                 enable_analytics: bool = True,
                 performance_monitoring: bool = True):
        """
        Initialize the selector manager
        
        Args:
            cache_file: File to cache selector performance data
            enable_analytics: Enable success rate tracking
            performance_monitoring: Enable performance optimization
        """
        self.cache_file = Path(cache_file) if cache_file else None
        self.enable_analytics = enable_analytics
        self.performance_monitoring = performance_monitoring
        
        # Initialize selector dictionaries
        self.selectors: Dict[SelectorType, Dict[str, SelectorInfo]] = defaultdict(dict)
        
        # Performance tracking
        self.selector_stats = defaultdict(lambda: {
            'total_time': 0.0,
            'total_calls': 0,
            'avg_time': 0.0,
            'success_rate': 0.0
        })
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize all selectors
        self._initialize_selectors()
        
        # Load cached performance data
        if self.cache_file and self.cache_file.exists():
            self._load_cache()
    
    def _initialize_selectors(self):
        """Initialize all CSS selectors for eBay jewelry extraction"""
        
        # ============ SEARCH RESULTS SELECTORS ============
        self._init_search_results_selectors()
        
        # ============ PRODUCT DETAILS SELECTORS ============
        self._init_product_details_selectors()
        
        # ============ JEWELRY-SPECIFIC SELECTORS ============
        self._init_jewelry_specific_selectors()
        
        # ============ IMAGES SELECTORS ============
        self._init_images_selectors()
        
        # ============ SHIPPING SELECTORS ============
        self._init_shipping_selectors()
        
        # ============ SELLER SELECTORS ============
        self._init_seller_selectors()
        
        # ============ MOBILE-SPECIFIC SELECTORS ============
        self._init_mobile_selectors()
    
    def _init_search_results_selectors(self):
        """Initialize search results page selectors"""
        
        # Listing containers
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_container'] = SelectorInfo(
            primary='div.s-item',
            fallbacks=[
                'div[data-view="mi:1686|iid:1"]',
                '.srp-results .s-item',
                'div.s-item__wrapper',
                'li.s-item'
            ],
            description="Main listing container in search results",
            performance_weight=1
        )
        
        # Listing titles
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_title'] = SelectorInfo(
            primary='h3.s-item__title',
            fallbacks=[
                '.s-item__title span[role="heading"]',
                '.s-item__title a',
                '.s-item__title',
                'h3.it-ttl a',
                '.lvtitle a'
            ],
            description="Product title in search results",
            performance_weight=1
        )
        
        # Listing prices
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_price'] = SelectorInfo(
            primary='span.s-item__price',
            fallbacks=[
                '.s-item__detail .s-item__price',
                '.notranslate',
                '.u-flL.condText+ .u-flL',
                '.amt.notranslate',
                '.u-flL .notranslate'
            ],
            description="Product price in search results",
            performance_weight=1
        )
        
        # Listing URLs
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_url'] = SelectorInfo(
            primary='h3.s-item__title a',
            fallbacks=[
                '.s-item__link',
                '.s-item__title a',
                'a.s-item__link',
                '.vip a'
            ],
            description="Product URL link in search results",
            performance_weight=1
        )
        
        # Listing images
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_image'] = SelectorInfo(
            primary='.s-item__image img',
            fallbacks=[
                '.s-item__wrapper img',
                'img.s-item__image',
                '.img img',
                '.s-item img'
            ],
            description="Product thumbnail image in search results",
            performance_weight=2
        )
        
        # Listing condition
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_condition'] = SelectorInfo(
            primary='.s-item__subtitle .SECONDARY_INFO',
            fallbacks=[
                '.s-item__condition-text',
                '.s-item__condition',
                '.condText',
                '.s-item__subtitle'
            ],
            description="Product condition in search results",
            performance_weight=3
        )
        
        # Shipping info in search results
        self.selectors[SelectorType.SEARCH_RESULTS]['listing_shipping'] = SelectorInfo(
            primary='.s-item__shipping',
            fallbacks=[
                '.s-item__logisticsCost',
                '.s-item__detail .s-item__shipping',
                '.s-item__freeXDays'
            ],
            description="Shipping information in search results",
            performance_weight=3
        )
    
    def _init_product_details_selectors(self):
        """Initialize product details page selectors"""
        
        # Main product title
        self.selectors[SelectorType.PRODUCT_DETAILS]['title'] = SelectorInfo(
            primary='h1#it-ttl',
            fallbacks=[
                'h1.x-item-title-label',
                '#ebayui-ellipsis-2',
                '.x-item-title-label',
                'h1.notranslate'
            ],
            description="Main product title on listing page",
            performance_weight=1
        )
        
        # Current price
        self.selectors[SelectorType.PRODUCT_DETAILS]['current_price'] = SelectorInfo(
            primary='.u-flL.condText+ .u-flL .notranslate',
            fallbacks=[
                '#prcIsum .notranslate',
                '.u-flL .notranslate',
                '.amt.notranslate',
                '#mm-saleDscPrc',
                '.cc-ts-BOLD'
            ],
            description="Current price on product page",
            performance_weight=1
        )
        
        # Original/was price
        self.selectors[SelectorType.PRODUCT_DETAILS]['original_price'] = SelectorInfo(
            primary='.u-flL.condText .notranslate',
            fallbacks=[
                '.originalRetailPrice .notranslate',
                '.wasPrice .notranslate',
                '#orgPrc .notranslate'
            ],
            description="Original/was price if item is on sale",
            performance_weight=2
        )
        
        # Product condition
        self.selectors[SelectorType.PRODUCT_DETAILS]['condition'] = SelectorInfo(
            primary='#u-flL.condText',
            fallbacks=[
                '.u-flL.condText',
                '.condText',
                '#vi-acc-del-range .u-flL'
            ],
            description="Product condition on listing page",
            performance_weight=2
        )
        
        # Item specifics section
        self.selectors[SelectorType.PRODUCT_DETAILS]['item_specifics'] = SelectorInfo(
            primary='.itemAttr',
            fallbacks=[
                '.viTabs.viTab.viTAbDiv.viTabs-selected .itemAttr',
                '#viTabs_0_is .itemAttr',
                'div.section .itemAttr'
            ],
            description="Item specifics/details section",
            performance_weight=2
        )
        
        # Description iframe
        self.selectors[SelectorType.PRODUCT_DETAILS]['description_iframe'] = SelectorInfo(
            primary='#desc_ifr',
            fallbacks=[
                'iframe#desc_ifr',
                '.d2-display iframe',
                '#viTabs_0_pd iframe'
            ],
            description="Description iframe container",
            performance_weight=3
        )
        
        # Watch/follow button
        self.selectors[SelectorType.PRODUCT_DETAILS]['watch_button'] = SelectorInfo(
            primary='#vi-acc-del-range .watchlink',
            fallbacks=[
                '.watchlink',
                '#vi-acc-del-range a[href*="watchitem"]'
            ],
            description="Watch/follow this item button",
            performance_weight=4
        )
    
    def _init_jewelry_specific_selectors(self):
        """Initialize jewelry-specific attribute selectors"""
        
        # Metal type
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['metal_type'] = SelectorInfo(
            primary='td:contains("Metal") + td',
            fallbacks=[
                'td:contains("Metal Type") + td',
                'td:contains("Base Metal") + td',
                'div:contains("Metal:") .attrValue',
                '[data-testid="ux-labels-values"] span:contains("Metal") + span'
            ],
            description="Jewelry metal type (gold, silver, etc.)",
            performance_weight=2
        )
        
        # Stone type
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['stone_type'] = SelectorInfo(
            primary='td:contains("Main Stone") + td',
            fallbacks=[
                'td:contains("Stone") + td',
                'td:contains("Gemstone") + td',
                'div:contains("Stone:") .attrValue',
                'td:contains("Center Stone") + td'
            ],
            description="Main stone/gemstone type",
            performance_weight=2
        )
        
        # Ring size
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['ring_size'] = SelectorInfo(
            primary='td:contains("Ring Size") + td',
            fallbacks=[
                'td:contains("Size") + td',
                'div:contains("Size:") .attrValue',
                'select[name="msku"] option[selected]',
                '.u-flL.vi-sMSku-gen select option[selected]'
            ],
            description="Ring size for ring jewelry",
            category_specific=JewelryCategory.RINGS,
            performance_weight=2
        )
        
        # Chain length (for necklaces/bracelets)
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['chain_length'] = SelectorInfo(
            primary='td:contains("Length") + td',
            fallbacks=[
                'td:contains("Chain Length") + td',
                'div:contains("Length:") .attrValue',
                'td:contains("Necklace Length") + td'
            ],
            description="Chain length for necklaces and bracelets",
            category_specific=JewelryCategory.NECKLACES,
            performance_weight=2
        )
        
        # Brand
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['brand'] = SelectorInfo(
            primary='td:contains("Brand") + td',
            fallbacks=[
                'td:contains("Manufacturer") + td',
                'div:contains("Brand:") .attrValue',
                'h2.it-ttl span:first-child',
                '.vi-acc-del-range .u-flL span'
            ],
            description="Jewelry brand/manufacturer",
            performance_weight=1
        )
        
        # Style
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['style'] = SelectorInfo(
            primary='td:contains("Style") + td',
            fallbacks=[
                'td:contains("Jewelry Style") + td',
                'div:contains("Style:") .attrValue',
                'td:contains("Type") + td'
            ],
            description="Jewelry style/type",
            performance_weight=2
        )
        
        # Setting type
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['setting_type'] = SelectorInfo(
            primary='td:contains("Setting") + td',
            fallbacks=[
                'td:contains("Setting Type") + td',
                'td:contains("Stone Setting") + td',
                'div:contains("Setting:") .attrValue'
            ],
            description="Stone setting type",
            performance_weight=3
        )
        
        # Carat weight
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['carat_weight'] = SelectorInfo(
            primary='td:contains("Carat") + td',
            fallbacks=[
                'td:contains("Total Carat Weight") + td',
                'td:contains("Diamond Weight") + td',
                'div:contains("Carat:") .attrValue'
            ],
            description="Stone carat weight",
            performance_weight=3
        )
        
        # Vintage/antique indicator
        self.selectors[SelectorType.JEWELRY_SPECIFIC]['vintage'] = SelectorInfo(
            primary='td:contains("Age") + td',
            fallbacks=[
                'td:contains("Era") + td',
                'td:contains("Time Period") + td',
                'div:contains("Age:") .attrValue'
            ],
            description="Vintage/antique age information",
            performance_weight=3
        )
    
    def _init_images_selectors(self):
        """Initialize image-related selectors"""
        
        # Main product image
        self.selectors[SelectorType.IMAGES]['main_image'] = SelectorInfo(
            primary='#icImg',
            fallbacks=[
                '#mainImgHldr img',
                '#PicturePanel img',
                '.img img',
                '.zoom_trigger img'
            ],
            description="Main product image",
            performance_weight=1
        )
        
        # Image gallery thumbnails
        self.selectors[SelectorType.IMAGES]['gallery_thumbnails'] = SelectorInfo(
            primary='.tdThumb img',
            fallbacks=[
                '#vi_main_img_fs .tdThumb img',
                '.thumb img',
                '.imgTbl img',
                '#PicturePanel .thumbnail img'
            ],
            description="Gallery thumbnail images",
            performance_weight=2
        )
        
        # Zoom/enlarged image
        self.selectors[SelectorType.IMAGES]['zoom_image'] = SelectorInfo(
            primary='#zoom_trigger',
            fallbacks=[
                '.zoom_trigger',
                '#mainImgHldr .zoom',
                '.enlarge'
            ],
            description="Zoom/enlarge image trigger",
            performance_weight=3
        )
        
        # High resolution image URLs
        self.selectors[SelectorType.IMAGES]['high_res_images'] = SelectorInfo(
            primary='script:contains("mainImgUrl")',
            fallbacks=[
                'script:contains("imageUrls")',
                'script:contains("originalImgUrl")',
                'link[rel="image_src"]'
            ],
            description="High resolution image URLs from JavaScript",
            performance_weight=4
        )
        
        # Image count indicator
        self.selectors[SelectorType.IMAGES]['image_count'] = SelectorInfo(
            primary='#vi_main_img_fs .imgCnt',
            fallbacks=[
                '.imgCnt',
                '#PicturePanel .imgCnt',
                '.zoom_pic_count'
            ],
            description="Total image count indicator",
            performance_weight=3
        )
    
    def _init_shipping_selectors(self):
        """Initialize shipping-related selectors"""
        
        # Shipping cost
        self.selectors[SelectorType.SHIPPING]['shipping_cost'] = SelectorInfo(
            primary='#fShippingCost .notranslate',
            fallbacks=[
                '.vi-price .notranslate',
                '#shipAddrAppr .vi-price .notranslate',
                '.shipping .notranslate',
                '.del-range .notranslate'
            ],
            description="Shipping cost amount",
            performance_weight=2
        )
        
        # Shipping location (ships from)
        self.selectors[SelectorType.SHIPPING]['ships_from'] = SelectorInfo(
            primary='#shipAddrAppr .u-flL',
            fallbacks=[
                '.del-range .u-flL',
                '#vi-acc-del-range .u-flL',
                '.shipsFromLocation'
            ],
            description="Item ships from location",
            performance_weight=2
        )
        
        # Delivery estimate
        self.selectors[SelectorType.SHIPPING]['delivery_estimate'] = SelectorInfo(
            primary='.vi-acc-del-range .vi-u-flR',
            fallbacks=[
                '.del-range .vi-u-flR',
                '.delivery-time',
                '.estimated-delivery'
            ],
            description="Estimated delivery timeframe",
            performance_weight=2
        )
        
        # International shipping
        self.selectors[SelectorType.SHIPPING]['international_shipping'] = SelectorInfo(
            primary='#internationalShipping',
            fallbacks=[
                '.international-shipping',
                '#shippingSection .international',
                'div:contains("International shipping")'
            ],
            description="International shipping availability",
            performance_weight=3
        )
        
        # Free shipping indicator
        self.selectors[SelectorType.SHIPPING]['free_shipping'] = SelectorInfo(
            primary='.vi-acc-del-range .notranslate:contains("Free")',
            fallbacks=[
                '.free-shipping',
                'span:contains("Free shipping")',
                '.del-range span:contains("Free")'
            ],
            description="Free shipping indicator",
            performance_weight=2
        )
        
        # Returns policy
        self.selectors[SelectorType.SHIPPING]['returns_policy'] = SelectorInfo(
            primary='#returnPolicy',
            fallbacks=[
                '.returns-policy',
                '#vi-ret-accrd-txt',
                'div:contains("Return policy")'
            ],
            description="Returns policy information",
            performance_weight=3
        )
    
    def _init_seller_selectors(self):
        """Initialize seller-related selectors"""
        
        # Seller name
        self.selectors[SelectorType.SELLER]['seller_name'] = SelectorInfo(
            primary='.mbg-nw',
            fallbacks=[
                '.it-sel a',
                '.seller-persona .seller-name',
                '#RtdMbr a',
                '.si-content a'
            ],
            description="Seller username/name",
            performance_weight=1
        )
        
        # Seller feedback score
        self.selectors[SelectorType.SELLER]['feedback_score'] = SelectorInfo(
            primary='.mbg-l a',
            fallbacks=[
                '.seller-persona .feedback-score',
                '#RtdMbr .mbg-l',
                '.si-content .mbg-l'
            ],
            description="Seller feedback score",
            performance_weight=2
        )
        
        # Seller feedback percentage
        self.selectors[SelectorType.SELLER]['feedback_percentage'] = SelectorInfo(
            primary='.mbg-l + .mbg-l',
            fallbacks=[
                '.feedback-percentage',
                '#RtdMbr .positive-feedback',
                '.si-content .positive-feedback'
            ],
            description="Seller positive feedback percentage",
            performance_weight=2
        )
        
        # Seller location
        self.selectors[SelectorType.SELLER]['seller_location'] = SelectorInfo(
            primary='.u-flL.condText .vi-acc-loc',
            fallbacks=[
                '.seller-location',
                '#RtdMbr .loc',
                '.si-content .loc',
                '.mbg-loc'
            ],
            description="Seller geographic location",
            performance_weight=2
        )
        
        # Top rated seller badge
        self.selectors[SelectorType.SELLER]['top_rated_badge'] = SelectorInfo(
            primary='.top-rated-seller',
            fallbacks=[
                '.trs-badge',
                'img[alt*="Top Rated"]',
                '.seller-badge'
            ],
            description="Top rated seller badge",
            performance_weight=3
        )
        
        # Store name
        self.selectors[SelectorType.SELLER]['store_name'] = SelectorInfo(
            primary='.str-seller-card .str-title',
            fallbacks=[
                '.store-name',
                '.str-name a',
                '#store_name'
            ],
            description="eBay store name if applicable",
            performance_weight=3
        )
        
        # Business seller indicator
        self.selectors[SelectorType.SELLER]['business_seller'] = SelectorInfo(
            primary='.business-seller',
            fallbacks=[
                'img[alt*="Business seller"]',
                '.biz-seller-badge',
                'span:contains("Business seller")'
            ],
            description="Business seller indicator",
            performance_weight=3
        )
    
    def _init_mobile_selectors(self):
        """Initialize mobile-specific selectors"""
        
        # Mobile listing container
        self.selectors[SelectorType.MOBILE_SPECIFIC]['mobile_listing'] = SelectorInfo(
            primary='.sresult',
            fallbacks=[
                '.mobile-listing',
                '.m-item',
                'div[data-view*="mobile"]'
            ],
            description="Mobile listing container",
            device_type=DeviceType.MOBILE,
            performance_weight=1
        )
        
        # Mobile title
        self.selectors[SelectorType.MOBILE_SPECIFIC]['mobile_title'] = SelectorInfo(
            primary='.sresult .title',
            fallbacks=[
                '.mobile-title',
                '.m-item-title',
                '.sresult h3'
            ],
            description="Mobile product title",
            device_type=DeviceType.MOBILE,
            performance_weight=1
        )
        
        # Mobile price
        self.selectors[SelectorType.MOBILE_SPECIFIC]['mobile_price'] = SelectorInfo(
            primary='.sresult .price',
            fallbacks=[
                '.mobile-price',
                '.m-item-price',
                '.sresult .amt'
            ],
            description="Mobile product price",
            device_type=DeviceType.MOBILE,
            performance_weight=1
        )
        
        # Mobile image
        self.selectors[SelectorType.MOBILE_SPECIFIC]['mobile_image'] = SelectorInfo(
            primary='.sresult img',
            fallbacks=[
                '.mobile-image img',
                '.m-item img',
                '.sresult .img img'
            ],
            description="Mobile product image",
            device_type=DeviceType.MOBILE,
            performance_weight=2
        )
    
    def get_selector(self, 
                    selector_type: SelectorType, 
                    selector_name: str,
                    device_type: DeviceType = DeviceType.DESKTOP,
                    category: Optional[JewelryCategory] = None) -> Optional[SelectorInfo]:
        """
        Get a selector with device and category awareness
        
        Args:
            selector_type: Type of selector needed
            selector_name: Specific selector name
            device_type: Target device type
            category: Jewelry category for specialized selectors
            
        Returns:
            SelectorInfo object or None if not found
        """
        try:
            # First try to get the specific selector
            if selector_name in self.selectors[selector_type]:
                selector = self.selectors[selector_type][selector_name]
                
                # Check if selector is appropriate for device type
                if selector.device_type in [device_type, DeviceType.ALL]:
                    # Check category specificity
                    if not selector.category_specific or selector.category_specific == category:
                        return selector
            
            # If mobile device and no mobile-specific selector, try mobile alternatives
            if device_type == DeviceType.MOBILE and selector_type != SelectorType.MOBILE_SPECIFIC:
                mobile_selector = self.get_selector(
                    SelectorType.MOBILE_SPECIFIC, 
                    f"mobile_{selector_name}",
                    DeviceType.MOBILE,
                    category
                )
                if mobile_selector:
                    return mobile_selector
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting selector {selector_type.value}.{selector_name}: {e}")
            return None
    
    def get_all_selectors(self, 
                         selector_name: str,
                         device_type: DeviceType = DeviceType.DESKTOP,
                         category: Optional[JewelryCategory] = None,
                         include_fallbacks: bool = True) -> List[str]:
        """
        Get all possible selectors for a given name (primary + fallbacks)
        
        Args:
            selector_name: Name of the selector
            device_type: Target device type
            category: Jewelry category
            include_fallbacks: Whether to include fallback selectors
            
        Returns:
            List of CSS selectors ordered by priority
        """
        all_selectors = []
        
        # Search through all selector types
        for selector_type in SelectorType:
            selector_info = self.get_selector(selector_type, selector_name, device_type, category)
            if selector_info:
                all_selectors.append(selector_info.primary)
                if include_fallbacks:
                    all_selectors.extend(selector_info.fallbacks)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_selectors = []
        for selector in all_selectors:
            if selector not in seen:
                seen.add(selector)
                unique_selectors.append(selector)
        
        return unique_selectors
    
    def record_success(self, 
                      selector_type: SelectorType, 
                      selector_name: str, 
                      selector_used: str, 
                      success: bool,
                      execution_time: float = 0.0):
        """
        Record the success/failure of a selector usage
        
        Args:
            selector_type: Type of selector used
            selector_name: Name of selector used
            selector_used: Actual CSS selector string used
            success: Whether extraction was successful
            execution_time: Time taken for extraction
        """
        if not self.enable_analytics:
            return
        
        try:
            selector_info = self.selectors[selector_type].get(selector_name)
            if selector_info:
                selector_info.total_attempts += 1
                selector_info.last_tested = datetime.now()
                
                if success:
                    selector_info.successful_extractions += 1
                
                # Update success rate
                selector_info.success_rate = (
                    selector_info.successful_extractions / selector_info.total_attempts
                )
                
                # Update performance stats
                if self.performance_monitoring:
                    key = f"{selector_type.value}.{selector_name}"
                    stats = self.selector_stats[key]
                    stats['total_calls'] += 1
                    stats['total_time'] += execution_time
                    stats['avg_time'] = stats['total_time'] / stats['total_calls']
                    stats['success_rate'] = selector_info.success_rate
        
        except Exception as e:
            self.logger.error(f"Error recording selector success: {e}")
    
    def get_optimized_selectors(self, 
                               selector_type: SelectorType,
                               min_success_rate: float = 0.8,
                               max_avg_time: float = 1.0) -> Dict[str, SelectorInfo]:
        """
        Get selectors that meet performance criteria
        
        Args:
            selector_type: Type of selectors to analyze
            min_success_rate: Minimum success rate threshold
            max_avg_time: Maximum average execution time
            
        Returns:
            Dictionary of optimized selectors
        """
        optimized = {}
        
        for name, selector_info in self.selectors[selector_type].items():
            if selector_info.success_rate >= min_success_rate:
                key = f"{selector_type.value}.{name}"
                avg_time = self.selector_stats[key]['avg_time']
                
                if avg_time <= max_avg_time or avg_time == 0.0:
                    optimized[name] = selector_info
        
        return optimized
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'total_selectors': sum(len(selectors) for selectors in self.selectors.values()),
            'selector_types': len(self.selectors),
            'performance_data': dict(self.selector_stats),
            'category_breakdown': {},
            'device_breakdown': {},
            'success_rates': {},
            'avg_execution_times': {}
        }
        
        # Category breakdown
        for category in JewelryCategory:
            count = 0
            for selectors in self.selectors.values():
                count += sum(1 for s in selectors.values() 
                           if s.category_specific == category)
            report['category_breakdown'][category.value] = count
        
        # Device breakdown
        for device in DeviceType:
            count = 0
            for selectors in self.selectors.values():
                count += sum(1 for s in selectors.values() 
                           if s.device_type == device)
            report['device_breakdown'][device.value] = count
        
        # Success rates and execution times
        for selector_type, selectors in self.selectors.items():
            type_name = selector_type.value
            report['success_rates'][type_name] = {}
            report['avg_execution_times'][type_name] = {}
            
            for name, selector_info in selectors.items():
                report['success_rates'][type_name][name] = selector_info.success_rate
                key = f"{type_name}.{name}"
                report['avg_execution_times'][type_name][name] = (
                    self.selector_stats[key]['avg_time']
                )
        
        return report
    
    def _load_cache(self):
        """Load cached selector performance data"""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Load selector statistics
            if 'selector_stats' in cache_data:
                self.selector_stats.update(cache_data['selector_stats'])
            
            # Load selector performance data
            if 'selectors' in cache_data:
                for selector_type_name, selectors_data in cache_data['selectors'].items():
                    selector_type = SelectorType(selector_type_name)
                    for name, selector_data in selectors_data.items():
                        if name in self.selectors[selector_type]:
                            # Update existing selector with cached data
                            cached_info = SelectorInfo.from_dict(selector_data)
                            current_info = self.selectors[selector_type][name]
                            
                            # Update performance metrics
                            current_info.success_rate = cached_info.success_rate
                            current_info.total_attempts = cached_info.total_attempts
                            current_info.successful_extractions = cached_info.successful_extractions
                            current_info.last_tested = cached_info.last_tested
            
            self.logger.info("Loaded selector performance cache")
            
        except Exception as e:
            self.logger.warning(f"Failed to load selector cache: {e}")
    
    def save_cache(self):
        """Save selector performance data to cache"""
        if not self.cache_file:
            return
        
        try:
            cache_data = {
                'last_updated': datetime.now().isoformat(),
                'selector_stats': dict(self.selector_stats),
                'selectors': {}
            }
            
            # Save selector data
            for selector_type, selectors in self.selectors.items():
                cache_data['selectors'][selector_type.value] = {}
                for name, selector_info in selectors.items():
                    cache_data['selectors'][selector_type.value][name] = selector_info.to_dict()
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info("Saved selector performance cache")
            
        except Exception as e:
            self.logger.error(f"Failed to save selector cache: {e}")
    
    def validate_selectors(self) -> Dict[str, Any]:
        """Validate all selectors for completeness and correctness"""
        validation_report = {
            'total_selectors': 0,
            'valid_selectors': 0,
            'invalid_selectors': [],
            'missing_fallbacks': [],
            'performance_issues': []
        }
        
        for selector_type, selectors in self.selectors.items():
            for name, selector_info in selectors.items():
                validation_report['total_selectors'] += 1
                
                # Basic validation
                if not selector_info.primary or not selector_info.description:
                    validation_report['invalid_selectors'].append(
                        f"{selector_type.value}.{name}: Missing primary selector or description"
                    )
                    continue
                
                # Check for fallbacks
                if len(selector_info.fallbacks) < 2:
                    validation_report['missing_fallbacks'].append(
                        f"{selector_type.value}.{name}: Insufficient fallback selectors"
                    )
                
                # Performance check
                if selector_info.success_rate > 0 and selector_info.success_rate < 0.5:
                    validation_report['performance_issues'].append(
                        f"{selector_type.value}.{name}: Low success rate ({selector_info.success_rate:.2f})"
                    )
                
                validation_report['valid_selectors'] += 1
        
        return validation_report
    
    def get_selector_count(self) -> Dict[str, int]:
        """Get count of selectors by type"""
        return {
            selector_type.value: len(selectors)
            for selector_type, selectors in self.selectors.items()
        }
    
    def get_categories_covered(self) -> List[str]:
        """Get list of jewelry categories with specific selectors"""
        categories = set()
        for selectors in self.selectors.values():
            for selector_info in selectors.values():
                if selector_info.category_specific:
                    categories.add(selector_info.category_specific.value)
        return sorted(list(categories))
    
    def get_reliability_features(self) -> Dict[str, Any]:
        """Get summary of reliability features"""
        total_primary = 0
        total_fallbacks = 0
        
        for selectors in self.selectors.values():
            for selector_info in selectors.values():
                total_primary += 1
                total_fallbacks += len(selector_info.fallbacks)
        
        return {
            'primary_selectors': total_primary,
            'fallback_selectors': total_fallbacks,
            'average_fallbacks_per_selector': total_fallbacks / total_primary if total_primary > 0 else 0,
            'success_rate_tracking': self.enable_analytics,
            'performance_monitoring': self.performance_monitoring,
            'device_responsive': True,
            'category_specific': True
        }