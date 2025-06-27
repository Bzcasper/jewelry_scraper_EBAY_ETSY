"""
Advanced Analytics Engine for Jewelry Database
Provides comprehensive analytics, reporting, and business intelligence capabilities.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .database_manager import DatabaseManager
from .models import JewelryCategory, JewelryMaterial

logger = logging.getLogger(__name__)


class AnalyticsType(str, Enum):
    """Types of analytics reports"""
    SUMMARY = "summary"
    CATEGORY_ANALYSIS = "category_analysis"
    PRICE_ANALYSIS = "price_analysis"
    SELLER_ANALYSIS = "seller_analysis"
    QUALITY_ANALYSIS = "quality_analysis"
    TREND_ANALYSIS = "trend_analysis"
    MARKET_INSIGHTS = "market_insights"
    PERFORMANCE_METRICS = "performance_metrics"


class TimeRange(str, Enum):
    """Time range options for analytics"""
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    ALL_TIME = "all_time"
    CUSTOM = "custom"


@dataclass
class AnalyticsRequest:
    """Analytics request configuration"""
    
    analytics_type: AnalyticsType
    time_range: TimeRange = TimeRange.MONTH
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Filters
    categories: Optional[List[str]] = None
    materials: Optional[List[str]] = None
    sellers: Optional[List[str]] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_quality_score: Optional[float] = None
    
    # Options
    include_charts: bool = False
    group_by: Optional[str] = None
    limit: Optional[int] = None
    cache_results: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsResult:
    """Analytics result container"""
    
    request: AnalyticsRequest
    data: Dict[str, Any]
    summary: Dict[str, Any]
    charts: Optional[Dict[str, Any]] = None
    generated_at: datetime = field(default_factory=datetime.now)
    cache_key: Optional[str] = None
    execution_time_ms: Optional[float] = None


class AnalyticsEngine:
    """
    Advanced analytics engine for jewelry database insights
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize analytics engine
        
        Args:
            database_manager: DatabaseManager instance
        """
        self.db_manager = database_manager
        self._cache_ttl_hours = 24  # Cache TTL in hours
        
        # Initialize analytics cache table
        self._ensure_cache_table()
    
    def _ensure_cache_table(self):
        """Ensure analytics cache table exists"""
        try:
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_cache (
                        cache_key TEXT PRIMARY KEY,
                        cache_value TEXT NOT NULL,
                        cache_type TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        hit_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to create analytics cache table: {e}")
    
    def generate_analytics(self, request: AnalyticsRequest) -> AnalyticsResult:
        """
        Generate analytics report based on request
        
        Args:
            request: AnalyticsRequest configuration
            
        Returns:
            AnalyticsResult with data and insights
        """
        start_time = datetime.now()
        
        # Generate cache key
        cache_key = self._generate_cache_key(request) if request.cache_results else None
        
        # Check cache first
        if cache_key:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Analytics cache hit for {request.analytics_type}")
                return cached_result
        
        try:
            # Generate analytics based on type
            if request.analytics_type == AnalyticsType.SUMMARY:
                data = self._generate_summary_analytics(request)
            elif request.analytics_type == AnalyticsType.CATEGORY_ANALYSIS:
                data = self._generate_category_analytics(request)
            elif request.analytics_type == AnalyticsType.PRICE_ANALYSIS:
                data = self._generate_price_analytics(request)
            elif request.analytics_type == AnalyticsType.SELLER_ANALYSIS:
                data = self._generate_seller_analytics(request)
            elif request.analytics_type == AnalyticsType.QUALITY_ANALYSIS:
                data = self._generate_quality_analytics(request)
            elif request.analytics_type == AnalyticsType.TREND_ANALYSIS:
                data = self._generate_trend_analytics(request)
            elif request.analytics_type == AnalyticsType.MARKET_INSIGHTS:
                data = self._generate_market_insights(request)
            elif request.analytics_type == AnalyticsType.PERFORMANCE_METRICS:
                data = self._generate_performance_metrics(request)
            else:
                raise ValueError(f"Unsupported analytics type: {request.analytics_type}")
            
            # Generate summary
            summary = self._generate_summary(data, request)
            
            # Generate charts if requested
            charts = None
            if request.include_charts and MATPLOTLIB_AVAILABLE:
                charts = self._generate_charts(data, request)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = AnalyticsResult(
                request=request,
                data=data,
                summary=summary,
                charts=charts,
                cache_key=cache_key,
                execution_time_ms=execution_time
            )
            
            # Cache result if requested
            if cache_key:
                self._cache_result(cache_key, result)
            
            logger.info(f"Generated {request.analytics_type} analytics in {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate analytics: {e}")
            raise
    
    def _get_date_filter(self, request: AnalyticsRequest) -> Tuple[Optional[str], List[Any]]:
        """Generate date filter SQL and parameters"""
        
        if request.time_range == TimeRange.CUSTOM:
            if request.start_date and request.end_date:
                return "scraped_at BETWEEN ? AND ?", [
                    request.start_date.isoformat(),
                    request.end_date.isoformat()
                ]
        else:
            now = datetime.now()
            
            if request.time_range == TimeRange.TODAY:
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                return "scraped_at >= ?", [start.isoformat()]
            
            elif request.time_range == TimeRange.WEEK:
                start = now - timedelta(days=7)
                return "scraped_at >= ?", [start.isoformat()]
            
            elif request.time_range == TimeRange.MONTH:
                start = now - timedelta(days=30)
                return "scraped_at >= ?", [start.isoformat()]
            
            elif request.time_range == TimeRange.QUARTER:
                start = now - timedelta(days=90)
                return "scraped_at >= ?", [start.isoformat()]
            
            elif request.time_range == TimeRange.YEAR:
                start = now - timedelta(days=365)
                return "scraped_at >= ?", [start.isoformat()]
        
        return None, []
    
    def _build_filters(self, request: AnalyticsRequest) -> Tuple[str, List[Any]]:
        """Build WHERE clause filters"""
        
        conditions = []
        params = []
        
        # Date filter
        date_condition, date_params = self._get_date_filter(request)
        if date_condition:
            conditions.append(date_condition)
            params.extend(date_params)
        
        # Category filter
        if request.categories:
            placeholders = ', '.join(['?' for _ in request.categories])
            conditions.append(f"category IN ({placeholders})")
            params.extend(request.categories)
        
        # Material filter
        if request.materials:
            placeholders = ', '.join(['?' for _ in request.materials])
            conditions.append(f"material IN ({placeholders})")
            params.extend(request.materials)
        
        # Seller filter
        if request.sellers:
            placeholders = ', '.join(['?' for _ in request.sellers])
            conditions.append(f"seller_name IN ({placeholders})")
            params.extend(request.sellers)
        
        # Price filters
        if request.min_price is not None:
            conditions.append("price >= ?")
            params.append(request.min_price)
        
        if request.max_price is not None:
            conditions.append("price <= ?")
            params.append(request.max_price)
        
        # Quality filter
        if request.min_quality_score is not None:
            conditions.append("data_completeness_score >= ?")
            params.append(request.min_quality_score)
        
        # Always filter for validated listings
        conditions.append("is_validated = ?")
        params.append(True)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        return where_clause, params
    
    def _generate_summary_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate comprehensive summary analytics"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Basic counts and metrics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_listings,
                    COUNT(DISTINCT category) as unique_categories,
                    COUNT(DISTINCT material) as unique_materials,
                    COUNT(DISTINCT seller_name) as unique_sellers,
                    COUNT(DISTINCT brand) as unique_brands,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(data_completeness_score) as avg_quality,
                    SUM(image_count) as total_images,
                    AVG(image_count) as avg_images_per_listing,
                    SUM(watchers) as total_watchers,
                    SUM(views) as total_views,
                    SUM(bids) as total_bids,
                    COUNT(CASE WHEN main_stone IS NOT NULL AND main_stone != '' THEN 1 END) as listings_with_gemstones
                FROM jewelry_listings
                WHERE {where_clause}
            """, params)
            
            summary_row = cursor.fetchone()
            summary_data = dict(summary_row) if summary_row else {}
            
            # Price distribution
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN price < 100 THEN 'Under $100'
                        WHEN price < 500 THEN '$100-$500'
                        WHEN price < 1000 THEN '$500-$1,000'
                        WHEN price < 5000 THEN '$1,000-$5,000'
                        ELSE 'Over $5,000'
                    END as price_range,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM jewelry_listings WHERE {where_clause}), 2) as percentage
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY price_range
                ORDER BY MIN(price)
            """, params * 2)
            
            price_distribution = [dict(row) for row in cursor.fetchall()]
            
            # Top categories
            cursor.execute(f"""
                SELECT category, COUNT(*) as count
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY category
                ORDER BY count DESC
                LIMIT 10
            """, params)
            
            top_categories = [dict(row) for row in cursor.fetchall()]
            
            # Top materials
            cursor.execute(f"""
                SELECT material, COUNT(*) as count
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY material
                ORDER BY count DESC
                LIMIT 10
            """, params)
            
            top_materials = [dict(row) for row in cursor.fetchall()]
            
            return {
                'overview': summary_data,
                'price_distribution': price_distribution,
                'top_categories': top_categories,
                'top_materials': top_materials
            }
    
    def _generate_category_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate category-specific analytics"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Category summary
            cursor.execute(f"""
                SELECT 
                    category,
                    COUNT(*) as listing_count,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(data_completeness_score) as avg_quality,
                    AVG(image_count) as avg_images,
                    AVG(watchers) as avg_watchers,
                    AVG(views) as avg_views,
                    COUNT(CASE WHEN main_stone IS NOT NULL AND main_stone != '' THEN 1 END) as with_gemstones,
                    COUNT(DISTINCT seller_name) as unique_sellers,
                    COUNT(DISTINCT brand) as unique_brands
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY category
                ORDER BY listing_count DESC
            """, params)
            
            category_summary = [dict(row) for row in cursor.fetchall()]
            
            # Category-material combinations
            cursor.execute(f"""
                SELECT 
                    category,
                    material,
                    COUNT(*) as count,
                    AVG(price) as avg_price
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY category, material
                HAVING COUNT(*) >= 3
                ORDER BY category, count DESC
            """, params)
            
            category_materials = [dict(row) for row in cursor.fetchall()]
            
            # Category price ranges
            cursor.execute(f"""
                SELECT 
                    category,
                    CASE 
                        WHEN price < 100 THEN 'Under $100'
                        WHEN price < 500 THEN '$100-$500'
                        WHEN price < 1000 THEN '$500-$1,000'
                        WHEN price < 5000 THEN '$1,000-$5,000'
                        ELSE 'Over $5,000'
                    END as price_range,
                    COUNT(*) as count
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY category, price_range
                ORDER BY category, MIN(price)
            """, params)
            
            category_price_ranges = [dict(row) for row in cursor.fetchall()]
            
            return {
                'category_summary': category_summary,
                'category_materials': category_materials,
                'category_price_ranges': category_price_ranges
            }
    
    def _generate_price_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate price-focused analytics"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Price statistics by category
            cursor.execute(f"""
                SELECT 
                    category,
                    COUNT(*) as count,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    CASE 
                        WHEN COUNT(*) >= 2 THEN
                            SQRT(SUM((price - (SELECT AVG(price) FROM jewelry_listings sub WHERE sub.category = jewelry_listings.category AND {where_clause})) * 
                                    (price - (SELECT AVG(price) FROM jewelry_listings sub WHERE sub.category = jewelry_listings.category AND {where_clause}))) / 
                                 (COUNT(*) - 1))
                        ELSE 0
                    END as price_stddev
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY category
                HAVING COUNT(*) >= 5
                ORDER BY avg_price DESC
            """, params * 2)
            
            price_by_category = [dict(row) for row in cursor.fetchall()]
            
            # Price statistics by material
            cursor.execute(f"""
                SELECT 
                    material,
                    COUNT(*) as count,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY material
                HAVING COUNT(*) >= 5
                ORDER BY avg_price DESC
            """, params)
            
            price_by_material = [dict(row) for row in cursor.fetchall()]
            
            # Most expensive listings
            cursor.execute(f"""
                SELECT 
                    listing_id,
                    title,
                    price,
                    currency,
                    category,
                    material,
                    brand,
                    seller_name,
                    watchers,
                    views
                FROM jewelry_listings
                WHERE {where_clause}
                ORDER BY price DESC
                LIMIT 20
            """, params)
            
            most_expensive = [dict(row) for row in cursor.fetchall()]
            
            # Price trends (if we have date data)
            cursor.execute(f"""
                SELECT 
                    DATE(scraped_at) as date,
                    COUNT(*) as listings_count,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY DATE(scraped_at)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            
            price_trends = [dict(row) for row in cursor.fetchall()]
            
            return {
                'price_by_category': price_by_category,
                'price_by_material': price_by_material,
                'most_expensive': most_expensive,
                'price_trends': price_trends
            }
    
    def _generate_seller_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate seller-focused analytics"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Top sellers by listing count
            cursor.execute(f"""
                SELECT 
                    seller_name,
                    COUNT(*) as listing_count,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(seller_rating) as avg_rating,
                    AVG(seller_feedback_count) as avg_feedback_count,
                    AVG(data_completeness_score) as avg_quality,
                    COUNT(DISTINCT category) as categories_sold,
                    AVG(watchers) as avg_watchers,
                    AVG(views) as avg_views
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY seller_name
                HAVING COUNT(*) >= 3
                ORDER BY listing_count DESC
                LIMIT 50
            """, params)
            
            top_sellers_by_count = [dict(row) for row in cursor.fetchall()]
            
            # Top sellers by total value
            cursor.execute(f"""
                SELECT 
                    seller_name,
                    COUNT(*) as listing_count,
                    SUM(price) as total_value,
                    AVG(price) as avg_price,
                    AVG(seller_rating) as avg_rating,
                    AVG(data_completeness_score) as avg_quality
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY seller_name
                HAVING COUNT(*) >= 3
                ORDER BY total_value DESC
                LIMIT 20
            """, params)
            
            top_sellers_by_value = [dict(row) for row in cursor.fetchall()]
            
            # Seller rating distribution
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN seller_rating >= 99 THEN '99-100%'
                        WHEN seller_rating >= 95 THEN '95-99%'
                        WHEN seller_rating >= 90 THEN '90-95%'
                        WHEN seller_rating >= 80 THEN '80-90%'
                        ELSE 'Under 80%'
                    END as rating_range,
                    COUNT(DISTINCT seller_name) as seller_count,
                    COUNT(*) as listing_count
                FROM jewelry_listings
                WHERE {where_clause} AND seller_rating IS NOT NULL
                GROUP BY rating_range
                ORDER BY MIN(seller_rating) DESC
            """, params)
            
            seller_rating_distribution = [dict(row) for row in cursor.fetchall()]
            
            return {
                'top_sellers_by_count': top_sellers_by_count,
                'top_sellers_by_value': top_sellers_by_value,
                'seller_rating_distribution': seller_rating_distribution
            }
    
    def _generate_quality_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate data quality analytics"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Quality score distribution
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN data_completeness_score >= 0.9 THEN 'Excellent (90-100%)'
                        WHEN data_completeness_score >= 0.7 THEN 'Good (70-89%)'
                        WHEN data_completeness_score >= 0.5 THEN 'Fair (50-69%)'
                        WHEN data_completeness_score >= 0.3 THEN 'Poor (30-49%)'
                        ELSE 'Very Poor (0-29%)'
                    END as quality_range,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM jewelry_listings WHERE {where_clause}), 2) as percentage
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY quality_range
                ORDER BY MIN(data_completeness_score) DESC
            """, params * 2)
            
            quality_distribution = [dict(row) for row in cursor.fetchall()]
            
            # Quality by category
            cursor.execute(f"""
                SELECT 
                    category,
                    COUNT(*) as count,
                    AVG(data_completeness_score) as avg_quality,
                    COUNT(CASE WHEN data_completeness_score >= 0.8 THEN 1 END) as high_quality_count,
                    ROUND(COUNT(CASE WHEN data_completeness_score >= 0.8 THEN 1 END) * 100.0 / COUNT(*), 2) as high_quality_percentage
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY category
                ORDER BY avg_quality DESC
            """, params)
            
            quality_by_category = [dict(row) for row in cursor.fetchall()]
            
            # Field completeness analysis
            cursor.execute(f"""
                SELECT 
                    COUNT(CASE WHEN title IS NOT NULL AND title != '' THEN 1 END) * 100.0 / COUNT(*) as title_completeness,
                    COUNT(CASE WHEN description IS NOT NULL AND description != '' THEN 1 END) * 100.0 / COUNT(*) as description_completeness,
                    COUNT(CASE WHEN brand IS NOT NULL AND brand != '' THEN 1 END) * 100.0 / COUNT(*) as brand_completeness,
                    COUNT(CASE WHEN main_stone IS NOT NULL AND main_stone != '' THEN 1 END) * 100.0 / COUNT(*) as gemstone_completeness,
                    COUNT(CASE WHEN size IS NOT NULL AND size != '' THEN 1 END) * 100.0 / COUNT(*) as size_completeness,
                    COUNT(CASE WHEN image_count > 0 THEN 1 END) * 100.0 / COUNT(*) as image_completeness,
                    COUNT(CASE WHEN seller_rating IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as seller_rating_completeness
                FROM jewelry_listings
                WHERE {where_clause}
            """, params)
            
            field_completeness = dict(cursor.fetchone())
            
            return {
                'quality_distribution': quality_distribution,
                'quality_by_category': quality_by_category,
                'field_completeness': field_completeness
            }
    
    def _generate_trend_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate trend analytics over time"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Daily scraping trends
            cursor.execute(f"""
                SELECT 
                    DATE(scraped_at) as date,
                    COUNT(*) as listings_count,
                    COUNT(DISTINCT seller_name) as unique_sellers,
                    AVG(price) as avg_price,
                    AVG(data_completeness_score) as avg_quality,
                    SUM(image_count) as total_images
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY DATE(scraped_at)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            
            daily_trends = [dict(row) for row in cursor.fetchall()]
            
            # Category trends
            cursor.execute(f"""
                SELECT 
                    DATE(scraped_at) as date,
                    category,
                    COUNT(*) as count,
                    AVG(price) as avg_price
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY DATE(scraped_at), category
                ORDER BY date DESC, count DESC
                LIMIT 100
            """, params)
            
            category_trends = [dict(row) for row in cursor.fetchall()]
            
            # Quality trends
            cursor.execute(f"""
                SELECT 
                    DATE(scraped_at) as date,
                    AVG(data_completeness_score) as avg_quality,
                    COUNT(CASE WHEN data_completeness_score >= 0.8 THEN 1 END) as high_quality_count,
                    COUNT(*) as total_count
                FROM jewelry_listings
                WHERE {where_clause}
                GROUP BY DATE(scraped_at)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            
            quality_trends = [dict(row) for row in cursor.fetchall()]
            
            return {
                'daily_trends': daily_trends,
                'category_trends': category_trends,
                'quality_trends': quality_trends
            }
    
    def _generate_market_insights(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate market insights and opportunities"""
        
        where_clause, params = self._build_filters(request)
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Hot categories (high engagement)
            cursor.execute(f"""
                SELECT 
                    category,
                    COUNT(*) as listing_count,
                    AVG(watchers) as avg_watchers,
                    AVG(views) as avg_views,
                    AVG(bids) as avg_bids,
                    AVG(price) as avg_price,
                    SUM(watchers + views + COALESCE(bids, 0)) as total_engagement
                FROM jewelry_listings
                WHERE {where_clause} AND (watchers > 0 OR views > 0 OR bids > 0)
                GROUP BY category
                HAVING COUNT(*) >= 10
                ORDER BY total_engagement DESC
                LIMIT 10
            """, params)
            
            hot_categories = [dict(row) for row in cursor.fetchall()]
            
            # Undervalued items (high engagement, low price)
            cursor.execute(f"""
                SELECT 
                    listing_id,
                    title,
                    price,
                    category,
                    material,
                    watchers,
                    views,
                    bids,
                    data_completeness_score,
                    (watchers + views + COALESCE(bids, 0)) / price as engagement_per_dollar
                FROM jewelry_listings
                WHERE {where_clause} 
                    AND (watchers > 0 OR views > 0 OR bids > 0)
                    AND price > 0
                ORDER BY engagement_per_dollar DESC
                LIMIT 20
            """, params)
            
            undervalued_items = [dict(row) for row in cursor.fetchall()]
            
            # Premium brands analysis
            cursor.execute(f"""
                SELECT 
                    brand,
                    COUNT(*) as listing_count,
                    AVG(price) as avg_price,
                    AVG(watchers) as avg_watchers,
                    AVG(data_completeness_score) as avg_quality,
                    COUNT(DISTINCT category) as categories
                FROM jewelry_listings
                WHERE {where_clause} 
                    AND brand IS NOT NULL 
                    AND brand != ''
                    AND price >= (SELECT AVG(price) FROM jewelry_listings WHERE {where_clause})
                GROUP BY brand
                HAVING COUNT(*) >= 5
                ORDER BY avg_price DESC
                LIMIT 15
            """, params * 2)
            
            premium_brands = [dict(row) for row in cursor.fetchall()]
            
            # Gemstone popularity
            cursor.execute(f"""
                SELECT 
                    main_stone,
                    COUNT(*) as listing_count,
                    AVG(price) as avg_price,
                    AVG(watchers) as avg_watchers,
                    COUNT(DISTINCT category) as categories
                FROM jewelry_listings
                WHERE {where_clause} 
                    AND main_stone IS NOT NULL 
                    AND main_stone != ''
                GROUP BY main_stone
                HAVING COUNT(*) >= 5
                ORDER BY listing_count DESC
                LIMIT 15
            """, params)
            
            gemstone_popularity = [dict(row) for row in cursor.fetchall()]
            
            return {
                'hot_categories': hot_categories,
                'undervalued_items': undervalued_items,
                'premium_brands': premium_brands,
                'gemstone_popularity': gemstone_popularity
            }
    
    def _generate_performance_metrics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate system performance metrics"""
        
        # Get database stats
        db_stats = self.db_manager.get_enhanced_database_stats()
        
        # Get scraping session performance
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Recent session performance
            cursor.execute("""
                SELECT 
                    AVG(CASE WHEN (listings_scraped + listings_failed) > 0 
                        THEN (listings_scraped * 100.0) / (listings_scraped + listings_failed) 
                        ELSE 0 END) as avg_success_rate,
                    AVG(average_quality_score) as avg_quality_score,
                    SUM(listings_scraped) as total_scraped,
                    SUM(listings_failed) as total_failed,
                    AVG(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL
                        THEN (JULIANDAY(completed_at) - JULIANDAY(started_at)) * 24 * 60
                        ELSE NULL END) as avg_duration_minutes
                FROM scraping_sessions
                WHERE started_at >= datetime('now', '-30 days')
            """)
            
            session_metrics = dict(cursor.fetchone())
            
            # Data quality trends
            cursor.execute("""
                SELECT 
                    AVG(data_completeness_score) as current_avg_quality,
                    COUNT(CASE WHEN data_completeness_score >= 0.8 THEN 1 END) * 100.0 / COUNT(*) as high_quality_percentage,
                    COUNT(CASE WHEN is_validated = 1 THEN 1 END) * 100.0 / COUNT(*) as validation_percentage
                FROM jewelry_listings
                WHERE scraped_at >= datetime('now', '-7 days')
            """)
            
            quality_metrics = dict(cursor.fetchone())
        
        return {
            'database_stats': db_stats._asdict() if hasattr(db_stats, '_asdict') else db_stats.__dict__,
            'session_metrics': session_metrics,
            'quality_metrics': quality_metrics
        }
    
    def _generate_summary(self, data: Dict[str, Any], request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate executive summary from analytics data"""
        
        summary = {
            'analytics_type': request.analytics_type.value,
            'time_range': request.time_range.value,
            'generated_at': datetime.now().isoformat(),
            'key_insights': []
        }
        
        if request.analytics_type == AnalyticsType.SUMMARY:
            overview = data.get('overview', {})
            summary['key_metrics'] = {
                'total_listings': overview.get('total_listings', 0),
                'avg_price': round(overview.get('avg_price', 0), 2),
                'avg_quality': round(overview.get('avg_quality', 0), 3),
                'total_sellers': overview.get('unique_sellers', 0)
            }
            
            # Generate insights
            if overview.get('total_listings', 0) > 1000:
                summary['key_insights'].append("Large dataset with over 1,000 listings")
            
            if overview.get('avg_quality', 0) > 0.8:
                summary['key_insights'].append("High average data quality (>80%)")
            elif overview.get('avg_quality', 0) < 0.5:
                summary['key_insights'].append("Data quality needs improvement (<50%)")
        
        return summary
    
    def _generate_charts(self, data: Dict[str, Any], request: AnalyticsRequest) -> Dict[str, Any]:
        """Generate charts for analytics data"""
        
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        charts = {}
        
        # This would generate actual charts - simplified for now
        # In a real implementation, you would generate matplotlib/seaborn charts
        # and save them as base64 encoded images or files
        
        return charts
    
    def _generate_cache_key(self, request: AnalyticsRequest) -> str:
        """Generate cache key for analytics request"""
        
        # Create a string representation of the request
        request_str = f"{request.analytics_type}|{request.time_range}|{request.start_date}|{request.end_date}|{request.categories}|{request.materials}|{request.min_price}|{request.max_price}|{request.min_quality_score}"
        
        # Generate MD5 hash
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[AnalyticsResult]:
        """Get cached analytics result"""
        
        try:
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if cache entry exists and is not expired
                cursor.execute("""
                    SELECT cache_value, hit_count
                    FROM analytics_cache
                    WHERE cache_key = ? AND expires_at > datetime('now')
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    # Update hit count and last accessed
                    cursor.execute("""
                        UPDATE analytics_cache 
                        SET hit_count = hit_count + 1, last_accessed = datetime('now')
                        WHERE cache_key = ?
                    """, (cache_key,))
                    conn.commit()
                    
                    # Deserialize result
                    cached_data = json.loads(row[0])
                    return AnalyticsResult(**cached_data)
        
        except Exception as e:
            logger.warning(f"Failed to get cached result: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: AnalyticsResult):
        """Cache analytics result"""
        
        try:
            # Calculate expiry time
            expires_at = datetime.now() + timedelta(hours=self._cache_ttl_hours)
            
            # Serialize result (excluding non-serializable fields)
            cache_data = {
                'request': result.request.__dict__,
                'data': result.data,
                'summary': result.summary,
                'generated_at': result.generated_at.isoformat(),
                'cache_key': result.cache_key,
                'execution_time_ms': result.execution_time_ms
            }
            
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO analytics_cache 
                    (cache_key, cache_value, cache_type, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    cache_key,
                    json.dumps(cache_data, default=str),
                    result.request.analytics_type.value,
                    expires_at.isoformat()
                ))
                
                conn.commit()
        
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def clear_cache(self, analytics_type: Optional[AnalyticsType] = None):
        """Clear analytics cache"""
        
        try:
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                if analytics_type:
                    cursor.execute("DELETE FROM analytics_cache WHERE cache_type = ?", (analytics_type.value,))
                else:
                    cursor.execute("DELETE FROM analytics_cache")
                
                conn.commit()
                logger.info(f"Cleared analytics cache for {analytics_type or 'all types'}")
        
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        try:
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as active_entries,
                        AVG(hit_count) as avg_hit_count,
                        MAX(hit_count) as max_hit_count,
                        cache_type,
                        COUNT(*) as type_count
                    FROM analytics_cache
                    GROUP BY cache_type
                """)
                
                cache_stats = [dict(row) for row in cursor.fetchall()]
                
                return {'cache_breakdown': cache_stats}
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


class ReportGenerator:
    """
    Report generator for creating formatted analytics reports
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
    
    def generate_dashboard_data(self, time_range: TimeRange = TimeRange.MONTH) -> Dict[str, Any]:
        """Generate data for analytics dashboard"""
        
        # Generate multiple analytics types for dashboard
        requests = [
            AnalyticsRequest(AnalyticsType.SUMMARY, time_range),
            AnalyticsRequest(AnalyticsType.CATEGORY_ANALYSIS, time_range),
            AnalyticsRequest(AnalyticsType.QUALITY_ANALYSIS, time_range),
            AnalyticsRequest(AnalyticsType.TREND_ANALYSIS, time_range)
        ]
        
        dashboard_data = {}
        
        for request in requests:
            result = self.analytics_engine.generate_analytics(request)
            dashboard_data[request.analytics_type.value] = {
                'data': result.data,
                'summary': result.summary
            }
        
        return dashboard_data
    
    def generate_market_report(self, time_range: TimeRange = TimeRange.MONTH) -> Dict[str, Any]:
        """Generate comprehensive market report"""
        
        requests = [
            AnalyticsRequest(AnalyticsType.MARKET_INSIGHTS, time_range),
            AnalyticsRequest(AnalyticsType.PRICE_ANALYSIS, time_range),
            AnalyticsRequest(AnalyticsType.CATEGORY_ANALYSIS, time_range),
            AnalyticsRequest(AnalyticsType.SELLER_ANALYSIS, time_range)
        ]
        
        report_data = {}
        
        for request in requests:
            result = self.analytics_engine.generate_analytics(request)
            report_data[request.analytics_type.value] = result.data
        
        return {
            'report_type': 'market_analysis',
            'time_range': time_range.value,
            'generated_at': datetime.now().isoformat(),
            'data': report_data
        }


# Convenience functions for common analytics scenarios
def get_category_insights(analytics_engine: AnalyticsEngine, 
                         category: str, 
                         time_range: TimeRange = TimeRange.MONTH) -> AnalyticsResult:
    """Get insights for specific category"""
    
    request = AnalyticsRequest(
        analytics_type=AnalyticsType.CATEGORY_ANALYSIS,
        time_range=time_range,
        categories=[category]
    )
    
    return analytics_engine.generate_analytics(request)


def get_price_analysis(analytics_engine: AnalyticsEngine,
                      min_price: float = None,
                      max_price: float = None,
                      time_range: TimeRange = TimeRange.MONTH) -> AnalyticsResult:
    """Get price analysis for specified range"""
    
    request = AnalyticsRequest(
        analytics_type=AnalyticsType.PRICE_ANALYSIS,
        time_range=time_range,
        min_price=min_price,
        max_price=max_price
    )
    
    return analytics_engine.generate_analytics(request)