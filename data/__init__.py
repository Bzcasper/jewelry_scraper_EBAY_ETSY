"""
Jewelry Scraper Data Module
High-performance SQLite database system for jewelry listings, images, and metadata.

This module provides comprehensive data management capabilities including:
- Advanced database schema with proper indexing
- High-performance CRUD operations with connection pooling  
- Sophisticated querying and filtering system
- Multi-format data export (CSV, JSON, XML, Excel)
- Automated data cleanup with retention policies
- Real-time statistics and analytics
- Robust backup and recovery system
- Data validation and integrity checks
- Database health monitoring and optimization

Usage:
    from src.jewelry_scraper.data import DatabaseManager, QueryBuilder, ExportManager
    
    # Initialize database manager with connection pooling
    db = DatabaseManager("jewelry_scraping.db", pool_size=10)
    
    # Advanced querying
    query = QueryBuilder().category("rings").price_range(100, 1000).build()
    results = db.query_listings(query)
    
    # Export data
    exporter = ExportManager(db)
    exporter.export_to_excel("jewelry_data.xlsx", include_images=True)
"""

from .database_manager import DatabaseManager, ConnectionPool
from .query_builder import QueryBuilder, QueryFilters, SortOptions
from .export_manager import ExportManager, ExportFormat
from .analytics_engine import AnalyticsEngine, ReportGenerator
from .backup_manager import BackupManager, BackupStrategy
from .validation_engine import ValidationEngine, IntegrityChecker
from ..models.jewelry_models import *
from .schema import JEWELRY_SCHEMA_SQL, JEWELRY_INDEXES_SQL, JEWELRY_VIEWS_SQL

__version__ = "1.0.0"
__author__ = "Jewelry Scraper Team"

# Export main classes
__all__ = [
    'DatabaseManager',
    'ConnectionPool',
    'QueryBuilder',
    'QueryFilters',
    'SortOptions',
    'ExportManager',
    'ExportFormat',
    'AnalyticsEngine',
    'ReportGenerator',
    'BackupManager',
    'BackupStrategy',
    'ValidationEngine',
    'IntegrityChecker',
    'JewelryListing',
    'JewelryImage',
    'JewelrySpecification',
    'ScrapingSession',
    'JEWELRY_SCHEMA_SQL',
    'JEWELRY_INDEXES_SQL',
    'JEWELRY_VIEWS_SQL'
]

# Module-level configuration
DEFAULT_DB_PATH = "jewelry_scraping.db"
DEFAULT_POOL_SIZE = 5
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CACHE_SIZE = 10000

# Performance settings
PERFORMANCE_CONFIG = {
    'wal_mode': True,
    'cache_size': DEFAULT_CACHE_SIZE,
    'temp_store': 'memory',
    'mmap_size': 268435456,  # 256MB
    'synchronous': 'normal',
    'journal_size_limit': 67108864,  # 64MB
    'auto_vacuum': 'incremental'
}

# Validation settings
VALIDATION_CONFIG = {
    'min_quality_score': 0.3,
    'required_fields': ['id', 'title', 'price', 'category', 'material'],
    'max_title_length': 500,
    'max_description_length': 10000,
    'validate_urls': True,
    'validate_images': True
}

# Export settings
EXPORT_CONFIG = {
    'max_export_size': 100000,  # Maximum records per export
    'include_metadata': False,
    'csv_delimiter': ',',
    'json_indent': 2,
    'excel_sheet_name': 'Jewelry Listings'
}

# Cleanup settings
CLEANUP_CONFIG = {
    'retention_days': 90,
    'cleanup_unvalidated': True,
    'cleanup_failed_images': True,
    'backup_before_cleanup': True
}
