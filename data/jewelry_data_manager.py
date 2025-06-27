"""
Comprehensive Jewelry Scraping Database Manager
High-performance SQLite data management system for jewelry listings, images, and metadata.

This module provides complete CRUD operations, query optimization, analytics, 
export functionality, and data validation for the jewelry scraping system.
"""

import sqlite3
import json
import csv
import os
import logging
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass
import pandas as pd

# Import the jewelry models
from ..models.jewelry_models import (
    JewelryListing, JewelryImage, JewelrySpecification, ScrapingSession,
    JewelryCategory, JewelryMaterial, ListingStatus, ScrapingStatus, ImageType,
    JEWELRY_SCHEMA_SQL, JEWELRY_INDEXES_SQL
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryFilters:
    """Data class for query filtering parameters"""
    category: Optional[str] = None
    material: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    brand: Optional[str] = None
    seller: Optional[str] = None
    condition: Optional[str] = None
    min_quality_score: Optional[float] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_images: Optional[bool] = None
    is_validated: Optional[bool] = None
    search_text: Optional[str] = None


@dataclass
class DatabaseStats:
    """Database statistics data class"""
    total_listings: int
    total_images: int
    total_specifications: int
    total_sessions: int
    avg_quality_score: float
    categories_breakdown: Dict[str, int]
    materials_breakdown: Dict[str, int]
    price_range: Tuple[float, float]
    recent_activity: Dict[str, int]
    storage_size_mb: float


class JewelryDatabaseManager:
    """
    Comprehensive database manager for jewelry scraping system
    Handles all database operations with performance optimization
    """

    def __init__(self, db_path: str = "jewelry_scraping.db", enable_wal: bool = True):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for better performance
        """
        self.db_path = db_path
        self.enable_wal = enable_wal
        self._ensure_database_exists()

        # Performance settings
        self.batch_size = 1000
        self.max_connections = 10

        logger.info(
            f"Initialized JewelryDatabaseManager with database: {db_path}")

    def _ensure_database_exists(self):
        """Ensure database file exists and create if needed"""
        if not os.path.exists(self.db_path):
            logger.info(f"Creating new database: {self.db_path}")
            self.initialize_database()
        else:
            logger.info(f"Using existing database: {self.db_path}")

    @contextmanager
    def get_connection(self, read_only: bool = False):
        """
        Context manager for database connections

        Args:
            read_only: Open connection in read-only mode
        """
        conn = None
        try:
            # Always use regular connection for now - read_only is for future optimization
            conn = sqlite3.connect(self.db_path)

            # Enable WAL mode for better performance
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")

            # Performance optimizations
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")

            # Row factory for named access
            conn.row_factory = sqlite3.Row

            yield conn

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_database(self) -> bool:
        """
        Initialize database with schema and indexes

        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create tables
                for table_name, schema_sql in JEWELRY_SCHEMA_SQL.items():
                    logger.info(f"Creating table: {table_name}")
                    cursor.execute(schema_sql)

                # Create indexes
                for index_sql in JEWELRY_INDEXES_SQL:
                    cursor.execute(index_sql)

                # Create additional performance views
                self._create_performance_views(cursor)

                conn.commit()
                logger.info("Database initialized successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def _create_performance_views(self, cursor):
        """Create database views for common queries"""

        # View for listing summaries
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS listing_summaries AS
            SELECT 
                listing_id,
                title,
                price,
                currency,
                category,
                material,
                brand,
                seller_name,
                data_completeness_score,
                image_count,
                scraped_at
            FROM jewelry_listings
            WHERE is_validated = TRUE
        """)

        # View for category statistics
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS category_stats AS
            SELECT 
                category,
                COUNT(*) as count,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(data_completeness_score) as avg_quality
            FROM jewelry_listings
            WHERE price > 0
            GROUP BY category
        """)

        # View for seller statistics
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS seller_stats AS
            SELECT 
                seller_name,
                COUNT(*) as listing_count,
                AVG(price) as avg_price,
                AVG(seller_rating) as avg_rating,
                AVG(data_completeness_score) as avg_quality
            FROM jewelry_listings
            WHERE seller_name IS NOT NULL
            GROUP BY seller_name
            HAVING COUNT(*) >= 5
        """)

        logger.info("Performance views created successfully")

    # === LISTING OPERATIONS ===

    def insert_listing(self, listing: JewelryListing) -> bool:
        """
        Insert a single jewelry listing

        Args:
            listing: JewelryListing instance

        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Validate listing before insertion
                if not listing.validate_for_database():
                    logger.warning(
                        f"Listing validation failed: {listing.validation_errors}")
                    return False

                # Update quality score
                listing.update_quality_score()

                # Prepare data for insertion
                listing_data = self._prepare_listing_data(listing)

                # Insert listing
                placeholders = ', '.join(['?' for _ in listing_data])
                columns = ', '.join(listing_data.keys())

                cursor.execute(f"""
                    INSERT OR REPLACE INTO jewelry_listings ({columns})
                    VALUES ({placeholders})
                """, list(listing_data.values()))

                conn.commit()
                logger.info(f"Inserted listing: {listing.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to insert listing {listing.id}: {e}")
            return False

    def batch_insert_listings(self, listings: List[JewelryListing]) -> int:
        """
        Batch insert multiple listings for better performance

        Args:
            listings: List of JewelryListing instances

        Returns:
            int: Number of successfully inserted listings
        """
        if not listings:
            return 0

        successful_inserts = 0

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Prepare batch data
                batch_data = []
                for listing in listings:
                    if listing.validate_for_database():
                        listing.update_quality_score()
                        batch_data.append(self._prepare_listing_data(listing))
                    else:
                        logger.warning(
                            f"Skipping invalid listing: {listing.id}")

                if not batch_data:
                    return 0

                # Get column names from first item
                columns = list(batch_data[0].keys())
                placeholders = ', '.join(['?' for _ in columns])
                columns_str = ', '.join(columns)

                # Batch insert
                cursor.executemany(f"""
                    INSERT OR REPLACE INTO jewelry_listings ({columns_str})
                    VALUES ({placeholders})
                """, [list(data.values()) for data in batch_data])

                successful_inserts = cursor.rowcount
                conn.commit()

                logger.info(f"Batch inserted {successful_inserts} listings")

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")

        return successful_inserts

    def _prepare_listing_data(self, listing: JewelryListing) -> Dict[str, Any]:
        """Prepare listing data for database insertion"""
        return {
            'listing_id': listing.id,
            'url': listing.listing_url,
            'title': listing.title,
            'price': listing.price,
            'original_price': listing.original_price,
            'currency': listing.currency,
            'condition': listing.condition,
            'availability': listing.availability,
            'seller_name': listing.seller_id,
            'seller_rating': listing.seller_rating,
            'seller_feedback_count': listing.seller_feedback_count,
            'category': listing.category.value,
            'subcategory': listing.subcategory,
            'brand': listing.brand,
            'material': listing.material.value,
            'materials': json.dumps(listing.materials) if listing.materials else None,
            'size': listing.size,
            'weight': listing.weight,
            'dimensions': listing.dimensions,
            'main_stone': listing.gemstone,
            'stone_color': listing.stone_color,
            'stone_clarity': listing.stone_clarity,
            'stone_cut': listing.stone_cut,
            'stone_carat': listing.stone_carat,
            'accent_stones': json.dumps(listing.accent_stones) if listing.accent_stones else None,
            'description': listing.description,
            'features': json.dumps(listing.features) if listing.features else None,
            'tags': json.dumps(listing.tags) if listing.tags else None,
            'item_number': listing.item_number,
            'listing_type': listing.listing_type,
            'listing_status': listing.listing_status.value,
            'watchers': listing.watchers,
            'views': listing.views,
            'bids': listing.bids,
            'time_left': listing.time_left,
            'shipping_cost': listing.shipping_cost,
            'ships_from': listing.ships_from,
            'ships_to': listing.ships_to,
            'image_count': listing.image_count,
            'description_length': listing.description_length,
            'data_completeness_score': listing.data_quality_score,
            'created_at': listing.created_at.isoformat(),
            'updated_at': listing.updated_at.isoformat(),
            'scraped_at': listing.scraped_at.isoformat(),
            'listing_date': listing.listing_date.isoformat() if listing.listing_date else None,
            'metadata': json.dumps(listing.metadata) if listing.metadata else None,
            'raw_data': json.dumps(listing.raw_data) if listing.raw_data else None,
            'is_validated': listing.is_validated,
            'validation_errors': json.dumps(listing.validation_errors) if listing.validation_errors else None
        }

    def get_listing(self, listing_id: str) -> Optional[JewelryListing]:
        """
        Retrieve a single listing by ID

        Args:
            listing_id: Unique listing identifier

        Returns:
            JewelryListing instance or None
        """
        try:
            with self.get_connection(read_only=True) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM jewelry_listings 
                    WHERE listing_id = ?
                """, (listing_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_listing(row)

        except Exception as e:
            logger.error(f"Failed to get listing {listing_id}: {e}")

        return None

    def _row_to_listing(self, row: sqlite3.Row) -> JewelryListing:
        """Convert database row to JewelryListing instance"""

        # Parse JSON fields
        materials = json.loads(row['materials']) if row['materials'] else []
        accent_stones = json.loads(
            row['accent_stones']) if row['accent_stones'] else []
        features = json.loads(row['features']) if row['features'] else []
        tags = json.loads(row['tags']) if row['tags'] else []
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        raw_data = json.loads(row['raw_data']) if row['raw_data'] else {}
        validation_errors = json.loads(
            row['validation_errors']) if row['validation_errors'] else []

        # Parse datetime fields
        created_at = datetime.fromisoformat(
            row['created_at']) if row['created_at'] else datetime.now()
        updated_at = datetime.fromisoformat(
            row['updated_at']) if row['updated_at'] else datetime.now()
        scraped_at = datetime.fromisoformat(
            row['scraped_at']) if row['scraped_at'] else datetime.now()
        listing_date = datetime.fromisoformat(
            row['listing_date']) if row['listing_date'] else None

        return JewelryListing(
            id=row['listing_id'],
            title=row['title'],
            price=row['price'],
            currency=row['currency'],
            condition=row['condition'],
            seller_id=row['seller_name'],
            listing_url=row['url'],
            end_time=datetime.fromisoformat(
                row['end_time']) if row['end_time'] else None,
            shipping_cost=row['shipping_cost'],
            category=JewelryCategory(row['category']),
            material=JewelryMaterial(row['material']),
            gemstone=row['main_stone'],
            size=row['size'],
            weight=row['weight'],
            brand=row['brand'],
            main_image_path=row['main_image_path'],
            original_price=row['original_price'],
            availability=row['availability'],
            seller_rating=row['seller_rating'],
            seller_feedback_count=row['seller_feedback_count'],
            subcategory=row['subcategory'],
            materials=materials,
            dimensions=row['dimensions'],
            stone_color=row['stone_color'],
            stone_clarity=row['stone_clarity'],
            stone_cut=row['stone_cut'],
            stone_carat=row['stone_carat'],
            accent_stones=accent_stones,
            description=row['description'],
            features=features,
            tags=tags,
            item_number=row['item_number'],
            listing_id=row['item_number'],
            listing_type=row['listing_type'],
            listing_status=ListingStatus(row['listing_status']),
            watchers=row['watchers'],
            views=row['views'],
            bids=row['bids'],
            time_left=row['time_left'],
            ships_from=row['ships_from'],
            ships_to=row['ships_to'],
            image_count=row['image_count'],
            description_length=row['description_length'],
            data_quality_score=row['data_completeness_score'],
            created_at=created_at,
            updated_at=updated_at,
            scraped_at=scraped_at,
            listing_date=listing_date,
            metadata=metadata,
            raw_data=raw_data,
            is_validated=bool(row['is_validated']),
            validation_errors=validation_errors,
            image_urls=[]  # Will be populated separately if needed
        )

    def query_listings(self, filters: QueryFilters, limit: int = 100, offset: int = 0) -> List[JewelryListing]:
        """
        Query listings with advanced filtering

        Args:
            filters: QueryFilters instance
            limit: Maximum number of results
            offset: Results offset for pagination

        Returns:
            List of JewelryListing instances
        """
        try:
            with self.get_connection(read_only=True) as conn:
                cursor = conn.cursor()

                # Build dynamic query
                query_parts = ["SELECT * FROM jewelry_listings WHERE 1=1"]
                params = []

                # Add filter conditions
                if filters.category:
                    query_parts.append("AND category = ?")
                    params.append(filters.category)

                if filters.material:
                    query_parts.append("AND material = ?")
                    params.append(filters.material)

                if filters.min_price is not None:
                    query_parts.append("AND price >= ?")
                    params.append(filters.min_price)

                if filters.max_price is not None:
                    query_parts.append("AND price <= ?")
                    params.append(filters.max_price)

                if filters.brand:
                    query_parts.append("AND brand LIKE ?")
                    params.append(f"%{filters.brand}%")

                if filters.seller:
                    query_parts.append("AND seller_name LIKE ?")
                    params.append(f"%{filters.seller}%")

                if filters.condition:
                    query_parts.append("AND condition = ?")
                    params.append(filters.condition)

                if filters.min_quality_score is not None:
                    query_parts.append("AND data_completeness_score >= ?")
                    params.append(filters.min_quality_score)

                if filters.date_from:
                    query_parts.append("AND scraped_at >= ?")
                    params.append(filters.date_from.isoformat())

                if filters.date_to:
                    query_parts.append("AND scraped_at <= ?")
                    params.append(filters.date_to.isoformat())

                if filters.has_images is not None:
                    if filters.has_images:
                        query_parts.append("AND image_count > 0")
                    else:
                        query_parts.append("AND image_count = 0")

                if filters.is_validated is not None:
                    query_parts.append("AND is_validated = ?")
                    params.append(filters.is_validated)

                if filters.search_text:
                    query_parts.append(
                        "AND (title LIKE ? OR description LIKE ?)")
                    search_term = f"%{filters.search_text}%"
                    params.extend([search_term, search_term])

                # Add ordering and pagination
                query_parts.append("ORDER BY scraped_at DESC")
                query_parts.append("LIMIT ? OFFSET ?")
                params.extend([limit, offset])

                # Execute query
                query = " ".join(query_parts)
                cursor.execute(query, params)

                # Convert rows to listings
                listings = []
                for row in cursor.fetchall():
                    listings.append(self._row_to_listing(row))

                return listings

        except Exception as e:
            logger.error(f"Failed to query listings: {e}")
            return []

    # === IMAGE OPERATIONS ===

    def insert_image(self, image: JewelryImage) -> bool:
        """Insert a single jewelry image record"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                image_data = {
                    'image_id': image.image_id,
                    'listing_id': image.listing_id,
                    'original_url': image.original_url,
                    'local_path': image.local_path,
                    'filename': image.filename,
                    'image_type': image.image_type.value,
                    'sequence_order': image.sequence_order,
                    'file_size': image.file_size,
                    'width': image.width,
                    'height': image.height,
                    'format': image.format,
                    'is_processed': image.is_processed,
                    'is_optimized': image.is_optimized,
                    'quality_score': image.quality_score,
                    'contains_text': image.contains_text,
                    'is_duplicate': image.is_duplicate,
                    'similarity_hash': image.similarity_hash,
                    'alt_text': image.alt_text,
                    'generated_description': image.generated_description,
                    'created_at': image.created_at.isoformat(),
                    'downloaded_at': image.downloaded_at.isoformat() if image.downloaded_at else None,
                    'processed_at': image.processed_at.isoformat() if image.processed_at else None,
                    'metadata': json.dumps(image.metadata) if image.metadata else None
                }

                columns = ', '.join(image_data.keys())
                placeholders = ', '.join(['?' for _ in image_data])

                cursor.execute(f"""
                    INSERT OR REPLACE INTO jewelry_images ({columns})
                    VALUES ({placeholders})
                """, list(image_data.values()))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to insert image {image.image_id}: {e}")
            return False

    def get_listing_images(self, listing_id: str) -> List[JewelryImage]:
        """Get all images for a specific listing"""
        try:
            with self.get_connection(read_only=True) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM jewelry_images 
                    WHERE listing_id = ?
                    ORDER BY sequence_order
                """, (listing_id,))

                images = []
                for row in cursor.fetchall():
                    images.append(self._row_to_image(row))

                return images

        except Exception as e:
            logger.error(f"Failed to get images for listing {listing_id}: {e}")
            return []

    def _row_to_image(self, row: sqlite3.Row) -> JewelryImage:
        """Convert database row to JewelryImage instance"""
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        created_at = datetime.fromisoformat(
            row['created_at']) if row['created_at'] else datetime.now()
        downloaded_at = datetime.fromisoformat(
            row['downloaded_at']) if row['downloaded_at'] else None
        processed_at = datetime.fromisoformat(
            row['processed_at']) if row['processed_at'] else None

        return JewelryImage(
            image_id=row['image_id'],
            listing_id=row['listing_id'],
            original_url=row['original_url'],
            local_path=row['local_path'],
            filename=row['filename'],
            image_type=ImageType(row['image_type']),
            sequence_order=row['sequence_order'],
            file_size=row['file_size'],
            width=row['width'],
            height=row['height'],
            format=row['format'],
            is_processed=bool(row['is_processed']),
            is_optimized=bool(row['is_optimized']),
            quality_score=row['quality_score'],
            contains_text=bool(row['contains_text']),
            is_duplicate=bool(row['is_duplicate']),
            similarity_hash=row['similarity_hash'],
            alt_text=row['alt_text'],
            generated_description=row['generated_description'],
            created_at=created_at,
            downloaded_at=downloaded_at,
            processed_at=processed_at,
            metadata=metadata
        )

    # === STATISTICS AND ANALYTICS ===

    def get_database_stats(self) -> DatabaseStats:
        """Get comprehensive database statistics"""
        try:
            with self.get_connection(read_only=True) as conn:
                cursor = conn.cursor()

                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM jewelry_listings")
                total_listings = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM jewelry_images")
                total_images = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM jewelry_specifications")
                total_specifications = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM scraping_sessions")
                total_sessions = cursor.fetchone()[0]

                # Average quality score
                cursor.execute(
                    "SELECT AVG(data_completeness_score) FROM jewelry_listings")
                avg_quality_score = cursor.fetchone()[0] or 0.0

                # Category breakdown
                cursor.execute(
                    "SELECT category, COUNT(*) FROM jewelry_listings GROUP BY category")
                categories_breakdown = dict(cursor.fetchall())

                # Materials breakdown
                cursor.execute(
                    "SELECT material, COUNT(*) FROM jewelry_listings GROUP BY material")
                materials_breakdown = dict(cursor.fetchall())

                # Price range
                cursor.execute(
                    "SELECT MIN(price), MAX(price) FROM jewelry_listings WHERE price > 0")
                price_result = cursor.fetchone()
                price_range = (price_result[0] or 0.0, price_result[1] or 0.0)

                # Recent activity (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("""
                    SELECT 
                        DATE(scraped_at) as date,
                        COUNT(*) as count
                    FROM jewelry_listings 
                    WHERE scraped_at >= ?
                    GROUP BY DATE(scraped_at)
                    ORDER BY date
                """, (week_ago,))
                recent_activity = dict(cursor.fetchall())

                # Database file size
                db_size_bytes = os.path.getsize(self.db_path)
                storage_size_mb = db_size_bytes / (1024 * 1024)

                return DatabaseStats(
                    total_listings=total_listings,
                    total_images=total_images,
                    total_specifications=total_specifications,
                    total_sessions=total_sessions,
                    avg_quality_score=avg_quality_score,
                    categories_breakdown=categories_breakdown,
                    materials_breakdown=materials_breakdown,
                    price_range=price_range,
                    recent_activity=recent_activity,
                    storage_size_mb=storage_size_mb
                )

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return DatabaseStats(0, 0, 0, 0, 0.0, {}, {}, (0.0, 0.0), {}, 0.0)

    # === EXPORT FUNCTIONALITY ===

    def export_to_csv(self, output_path: str, filters: Optional[QueryFilters] = None) -> bool:
        """Export listings to CSV format"""
        try:
            filters = filters or QueryFilters()
            listings = self.query_listings(
                filters, limit=10000)  # Large limit for export

            if not listings:
                logger.warning("No listings found for export")
                return False

            # Convert to dictionaries
            data = []
            for listing in listings:
                listing_dict = listing.to_dict(include_metadata=False)
                # Flatten complex fields
                listing_dict['materials'] = ', '.join(
                    listing.materials) if listing.materials else ''
                listing_dict['features'] = ', '.join(
                    listing.features) if listing.features else ''
                listing_dict['tags'] = ', '.join(
                    listing.tags) if listing.tags else ''
                data.append(listing_dict)

            # Write to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(listings)} listings to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False

    def export_to_json(self, output_path: str, filters: Optional[QueryFilters] = None,
                       include_metadata: bool = True) -> bool:
        """Export listings to JSON format"""
        try:
            filters = filters or QueryFilters()
            listings = self.query_listings(filters, limit=10000)

            if not listings:
                logger.warning("No listings found for export")
                return False

            # Convert to dictionaries
            data = [listing.to_dict(include_metadata=include_metadata)
                    for listing in listings]

            # Write to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Exported {len(listings)} listings to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False

    # === DATA CLEANUP ===

    def cleanup_old_data(self, days_old: int = 30, dry_run: bool = True) -> Dict[str, int]:
        """
        Clean up old data based on retention policy

        Args:
            days_old: Data older than this many days will be cleaned
            dry_run: If True, only count what would be deleted

        Returns:
            Dict with cleanup statistics
        """
        try:
            cutoff_date = (datetime.now() -
                           timedelta(days=days_old)).isoformat()

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Count items to be deleted
                cursor.execute("""
                    SELECT COUNT(*) FROM jewelry_listings 
                    WHERE scraped_at < ? AND is_validated = FALSE
                """, (cutoff_date,))
                listings_count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(*) FROM jewelry_images 
                    WHERE created_at < ? AND listing_id IN (
                        SELECT listing_id FROM jewelry_listings 
                        WHERE scraped_at < ? AND is_validated = FALSE
                    )
                """, (cutoff_date, cutoff_date))
                images_count = cursor.fetchone()[0]

                results = {
                    'listings_to_delete': listings_count,
                    'images_to_delete': images_count,
                    'dry_run': dry_run
                }

                if not dry_run and (listings_count > 0 or images_count > 0):
                    # Delete old unvalidated listings and their images
                    cursor.execute("""
                        DELETE FROM jewelry_listings 
                        WHERE scraped_at < ? AND is_validated = FALSE
                    """, (cutoff_date,))

                    # Foreign key constraints will cascade delete images
                    conn.commit()

                    logger.info(
                        f"Cleaned up {listings_count} old listings and {images_count} images")
                    results['deleted'] = True
                else:
                    results['deleted'] = False

                return results

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {'error': str(e)}

    # === BACKUP AND RECOVERY ===

    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a database backup

        Args:
            backup_path: Custom backup path (optional)

        Returns:
            str: Path to backup file
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"jewelry_scraping_backup_{timestamp}.db"

            # Create backup using file copy
            shutil.copy2(self.db_path, backup_path)

            # Verify backup
            with sqlite3.connect(backup_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM jewelry_listings")
                count = cursor.fetchone()[0]

            logger.info(f"Created backup with {count} listings: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore database from backup

        Args:
            backup_path: Path to backup file

        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Verify backup integrity
            with sqlite3.connect(backup_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]

                if integrity != "ok":
                    logger.error(f"Backup integrity check failed: {integrity}")
                    return False

            # Create current backup before restore
            current_backup = self.create_backup()
            logger.info(f"Created current database backup: {current_backup}")

            # Restore from backup
            shutil.copy2(backup_path, self.db_path)

            logger.info(f"Restored database from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

    # === VALIDATION ===

    def validate_database_integrity(self) -> Dict[str, Any]:
        """
        Comprehensive database integrity validation

        Returns:
            Dict with validation results
        """
        results = {
            'integrity_check': False,
            'foreign_key_check': False,
            'orphaned_images': 0,
            'orphaned_specs': 0,
            'duplicate_listings': 0,
            'missing_required_fields': 0,
            'errors': []
        }

        try:
            with self.get_connection(read_only=True) as conn:
                cursor = conn.cursor()

                # SQLite integrity check
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                results['integrity_check'] = integrity_result == "ok"

                if not results['integrity_check']:
                    results['errors'].append(
                        f"Integrity check failed: {integrity_result}")

                # Foreign key constraints check
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                results['foreign_key_check'] = len(fk_violations) == 0

                if fk_violations:
                    results['errors'].extend(
                        [f"FK violation: {v}" for v in fk_violations])

                # Check for orphaned images
                cursor.execute("""
                    SELECT COUNT(*) FROM jewelry_images 
                    WHERE listing_id NOT IN (SELECT listing_id FROM jewelry_listings)
                """)
                results['orphaned_images'] = cursor.fetchone()[0]

                # Check for orphaned specifications
                cursor.execute("""
                    SELECT COUNT(*) FROM jewelry_specifications 
                    WHERE listing_id NOT IN (SELECT listing_id FROM jewelry_listings)
                """)
                results['orphaned_specs'] = cursor.fetchone()[0]

                # Check for duplicate listings
                cursor.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT url, COUNT(*) 
                        FROM jewelry_listings 
                        GROUP BY url 
                        HAVING COUNT(*) > 1
                    )
                """)
                results['duplicate_listings'] = cursor.fetchone()[0]

                # Check for missing required fields
                cursor.execute("""
                    SELECT COUNT(*) FROM jewelry_listings 
                    WHERE title IS NULL OR title = '' 
                       OR price IS NULL OR price <= 0
                       OR seller_name IS NULL OR seller_name = ''
                       OR category IS NULL OR category = ''
                       OR material IS NULL OR material = ''
                """)
                results['missing_required_fields'] = cursor.fetchone()[0]

                logger.info(
                    f"Database validation completed. Issues found: {len(results['errors'])}")

        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            results['errors'].append(str(e))

        return results

    # === UTILITY METHODS ===

    def optimize_database(self) -> bool:
        """Optimize database performance"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Update statistics
                cursor.execute("ANALYZE")

                # Vacuum to reclaim space
                cursor.execute("VACUUM")

                # Optimize
                cursor.execute("PRAGMA optimize")

                conn.commit()
                logger.info("Database optimization completed")
                return True

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False

    def get_table_sizes(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        try:
            with self.get_connection(read_only=True) as conn:
                cursor = conn.cursor()

                tables = ['jewelry_listings', 'jewelry_images',
                          'jewelry_specifications', 'scraping_sessions']
                sizes = {}

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    sizes[table] = cursor.fetchone()[0]

                return sizes

        except Exception as e:
            logger.error(f"Failed to get table sizes: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize database manager
    db_manager = JewelryDatabaseManager("jewelry_scraping.db")

    # Get database statistics
    stats = db_manager.get_database_stats()
    print(f"Database Statistics:")
    print(f"Total Listings: {stats.total_listings}")
    print(f"Total Images: {stats.total_images}")
    print(f"Average Quality Score: {stats.avg_quality_score:.2f}")
    print(f"Storage Size: {stats.storage_size_mb:.2f} MB")

    # Example query
    filters = QueryFilters(
        category="rings",
        min_price=100.0,
        max_price=1000.0,
        min_quality_score=0.7
    )

    listings = db_manager.query_listings(filters, limit=10)
    print(f"\nFound {len(listings)} listings matching criteria")
