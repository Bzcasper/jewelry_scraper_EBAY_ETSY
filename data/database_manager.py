"""
Enhanced Database Manager with Connection Pooling and Performance Optimization
High-performance SQLite database management for jewelry scraping system.
"""

import sqlite3
import json
import os
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union, ContextManager
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue, Empty
import atexit

from .models import JewelryListing, JewelryImage, JewelrySpecification, ScrapingSession
from .schema import JEWELRY_SCHEMA_SQL, JEWELRY_INDEXES_SQL, JEWELRY_VIEWS_SQL

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Thread-safe SQLite connection pool for improved performance
    """
    
    def __init__(self, db_path: str, pool_size: int = 5, max_connections: int = 20):
        self.db_path = db_path
        self.pool_size = pool_size
        self.max_connections = max_connections
        self._pool = Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = threading.Lock()
        
        # Initialize pool with minimum connections
        self._initialize_pool()
        
        # Register cleanup on exit
        atexit.register(self._cleanup_pool)
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections"""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection with optimal settings"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Apply performance optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Row factory for named access
            conn.row_factory = sqlite3.Row
            
            with self._lock:
                self._created_connections += 1
            
            logger.debug(f"Created new database connection (total: {self._created_connections})")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self, timeout: float = 5.0) -> ContextManager[sqlite3.Connection]:
        """
        Get a connection from the pool
        
        Args:
            timeout: Maximum time to wait for a connection
            
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        start_time = time.time()
        
        try:
            # Try to get connection from pool
            try:
                conn = self._pool.get(timeout=timeout)
            except Empty:
                # Pool is empty, create new connection if under limit
                if self._created_connections < self.max_connections:
                    conn = self._create_connection()
                    if not conn:
                        raise Exception("Failed to create new connection")
                else:
                    raise Exception(f"Connection pool exhausted (max: {self.max_connections})")
            
            # Test connection
            try:
                conn.execute("SELECT 1").fetchone()
            except sqlite3.Error:
                # Connection is stale, create new one
                conn.close()
                conn = self._create_connection()
                if not conn:
                    raise Exception("Failed to create replacement connection")
            
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            # Return connection to pool
            if conn:
                try:
                    # Reset connection state
                    conn.rollback()
                    self._pool.put(conn, timeout=1.0)
                except (Empty, sqlite3.Error):
                    # Pool is full or connection is bad, close it
                    try:
                        conn.close()
                        with self._lock:
                            self._created_connections -= 1
                    except:
                        pass
    
    def _cleanup_pool(self):
        """Clean up all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'pool_size': self.pool_size,
            'max_connections': self.max_connections,
            'created_connections': self._created_connections,
            'available_connections': self._pool.qsize(),
            'active_connections': self._created_connections - self._pool.qsize()
        }


@dataclass
class DatabaseStats:
    """Enhanced database statistics"""
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
    connection_pool_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class DatabaseManager:
    """
    Enhanced database manager with connection pooling and advanced features
    """
    
    def __init__(self, db_path: str = "jewelry_scraping.db", pool_size: int = 5, enable_wal: bool = True):
        """
        Initialize enhanced database manager
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Connection pool size
            enable_wal: Enable WAL mode for better performance
        """
        self.db_path = Path(db_path).resolve()
        self.pool_size = pool_size
        self.enable_wal = enable_wal
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        self.connection_pool = ConnectionPool(str(self.db_path), pool_size)
        
        # Initialize database if needed
        self._ensure_database_exists()
        
        # Performance tracking
        self._performance_metrics = {
            'queries_executed': 0,
            'total_query_time': 0.0,
            'inserts_performed': 0,
            'exports_completed': 0,
            'last_optimization': None
        }
        
        logger.info(f"Enhanced DatabaseManager initialized: {self.db_path}")
    
    def _ensure_database_exists(self):
        """Ensure database exists and is properly initialized"""
        if not self.db_path.exists():
            logger.info(f"Creating new database: {self.db_path}")
            self.initialize_database()
        else:
            logger.info(f"Using existing database: {self.db_path}")
            self._verify_schema()
    
    def _verify_schema(self):
        """Verify database schema is up to date"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if main tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('jewelry_listings', 'jewelry_images', 'jewelry_specifications', 'scraping_sessions')
                """)
                
                existing_tables = {row[0] for row in cursor.fetchall()}
                required_tables = set(JEWELRY_SCHEMA_SQL.keys())
                
                if not required_tables.issubset(existing_tables):
                    logger.info("Schema verification failed, reinitializing database")
                    self.initialize_database()
                else:
                    logger.info("Database schema verified successfully")
                    
        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            self.initialize_database()
    
    def initialize_database(self) -> bool:
        """
        Initialize database with enhanced schema and indexes
        
        Returns:
            bool: Success status
        """
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables
                for table_name, schema_sql in JEWELRY_SCHEMA_SQL.items():
                    logger.info(f"Creating table: {table_name}")
                    cursor.execute(schema_sql)
                
                # Create indexes for performance
                for index_sql in JEWELRY_INDEXES_SQL:
                    cursor.execute(index_sql)
                
                # Create performance views
                for view_sql in JEWELRY_VIEWS_SQL:
                    cursor.execute(view_sql)
                
                # Create additional performance optimizations
                self._create_advanced_indexes(cursor)
                
                conn.commit()
                logger.info("Database initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _create_advanced_indexes(self, cursor):
        """Create advanced composite indexes for better query performance"""
        
        advanced_indexes = [
            # Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_listings_category_price ON jewelry_listings(category, price)",
            "CREATE INDEX IF NOT EXISTS idx_listings_material_category ON jewelry_listings(material, category)",
            "CREATE INDEX IF NOT EXISTS idx_listings_seller_status ON jewelry_listings(seller_name, listing_status)",
            "CREATE INDEX IF NOT EXISTS idx_listings_quality_scraped ON jewelry_listings(data_completeness_score, scraped_at)",
            
            # Full-text search indexes
            "CREATE INDEX IF NOT EXISTS idx_listings_title_fts ON jewelry_listings(title)",
            "CREATE INDEX IF NOT EXISTS idx_listings_description_fts ON jewelry_listings(description)",
            
            # Image-related composite indexes
            "CREATE INDEX IF NOT EXISTS idx_images_listing_type ON jewelry_images(listing_id, image_type)",
            "CREATE INDEX IF NOT EXISTS idx_images_processed_type ON jewelry_images(is_processed, image_type)",
            
            # Session performance indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_status_started ON scraping_sessions(status, started_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_query_status ON scraping_sessions(search_query, status)"
        ]
        
        for index_sql in advanced_indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                logger.warning(f"Failed to create advanced index: {e}")
    
    # === ENHANCED LISTING OPERATIONS ===
    
    def insert_listing(self, listing: JewelryListing) -> bool:
        """
        Insert a single jewelry listing with enhanced validation
        
        Args:
            listing: JewelryListing instance
            
        Returns:
            bool: Success status
        """
        start_time = time.time()
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Enhanced validation
                if not listing.validate_for_database():
                    logger.warning(f"Listing validation failed: {listing.validation_errors}")
                    return False
                
                # Update quality score
                listing.update_quality_score()
                
                # Prepare data for insertion
                listing_data = self._prepare_listing_data(listing)
                
                # Insert with UPSERT for better performance
                columns = ', '.join(listing_data.keys())
                placeholders = ', '.join([f':{key}' for key in listing_data.keys()])
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO jewelry_listings ({columns})
                    VALUES ({placeholders})
                """, listing_data)
                
                conn.commit()
                
                # Update performance metrics
                self._performance_metrics['inserts_performed'] += 1
                self._performance_metrics['total_query_time'] += time.time() - start_time
                
                logger.debug(f"Inserted listing: {listing.id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert listing {listing.id}: {e}")
            return False
    
    def batch_insert_listings(self, listings: List[JewelryListing], batch_size: int = 1000) -> int:
        """
        Enhanced batch insert with chunking and transaction optimization
        
        Args:
            listings: List of JewelryListing instances
            batch_size: Number of records per batch
            
        Returns:
            int: Number of successfully inserted listings
        """
        if not listings:
            return 0
        
        successful_inserts = 0
        start_time = time.time()
        
        try:
            # Process in chunks for memory efficiency
            for i in range(0, len(listings), batch_size):
                chunk = listings[i:i + batch_size]
                
                with self.connection_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Prepare batch data
                    batch_data = []
                    for listing in chunk:
                        if listing.validate_for_database():
                            listing.update_quality_score()
                            batch_data.append(self._prepare_listing_data(listing))
                        else:
                            logger.warning(f"Skipping invalid listing: {listing.id}")
                    
                    if not batch_data:
                        continue
                    
                    # Get column names from first item
                    columns = list(batch_data[0].keys())
                    placeholders = ', '.join([f':{key}' for key in columns])
                    columns_str = ', '.join(columns)
                    
                    # Batch insert with transaction
                    cursor.executemany(f"""
                        INSERT OR REPLACE INTO jewelry_listings ({columns_str})
                        VALUES ({placeholders})
                    """, batch_data)
                    
                    successful_inserts += len(batch_data)
                    conn.commit()
                    
                    logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_data)} listings")
            
            # Update performance metrics
            self._performance_metrics['inserts_performed'] += successful_inserts
            self._performance_metrics['total_query_time'] += time.time() - start_time
            
            logger.info(f"Batch inserted {successful_inserts} listings total")
                
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
        
        return successful_inserts
    
    def _prepare_listing_data(self, listing: JewelryListing) -> Dict[str, Any]:
        """Enhanced listing data preparation with better JSON handling"""
        
        # Helper function to safely serialize JSON
        def safe_json_dumps(value):
            if value is None:
                return None
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return json.dumps(str(value))
        
        return {
            'listing_id': listing.id,
            'url': listing.listing_url,
            'title': listing.title,
            'price': listing.price,
            'original_price': listing.original_price,
            'currency': listing.currency,
            'condition': listing.condition,
            'availability': getattr(listing, 'availability', None),
            'seller_name': listing.seller_id,
            'seller_rating': getattr(listing, 'seller_rating', None),
            'seller_feedback_count': getattr(listing, 'seller_feedback_count', None),
            'category': listing.category.value,
            'subcategory': getattr(listing, 'subcategory', None),
            'brand': getattr(listing, 'brand', None),
            'material': listing.material.value,
            'materials': safe_json_dumps(getattr(listing, 'materials', [])),
            'size': getattr(listing, 'size', None),
            'weight': getattr(listing, 'weight', None),
            'dimensions': getattr(listing, 'dimensions', None),
            'main_stone': getattr(listing, 'gemstone', None),
            'stone_color': getattr(listing, 'stone_color', None),
            'stone_clarity': getattr(listing, 'stone_clarity', None),
            'stone_cut': getattr(listing, 'stone_cut', None),
            'stone_carat': getattr(listing, 'stone_carat', None),
            'accent_stones': safe_json_dumps(getattr(listing, 'accent_stones', [])),
            'description': getattr(listing, 'description', None),
            'features': safe_json_dumps(getattr(listing, 'features', [])),
            'tags': safe_json_dumps(getattr(listing, 'tags', [])),
            'item_number': getattr(listing, 'item_number', None),
            'listing_type': getattr(listing, 'listing_type', None),
            'listing_status': getattr(listing, 'listing_status', 'unknown'),
            'watchers': getattr(listing, 'watchers', None),
            'views': getattr(listing, 'views', None),
            'bids': getattr(listing, 'bids', None),
            'time_left': getattr(listing, 'time_left', None),
            'shipping_cost': getattr(listing, 'shipping_cost', None),
            'ships_from': getattr(listing, 'ships_from', None),
            'ships_to': getattr(listing, 'ships_to', None),
            'image_count': getattr(listing, 'image_count', len(listing.image_urls)),
            'description_length': getattr(listing, 'description_length', len(getattr(listing, 'description', '') or '')),
            'data_completeness_score': listing.data_quality_score,
            'created_at': listing.created_at.isoformat(),
            'updated_at': listing.updated_at.isoformat(),
            'scraped_at': listing.scraped_at.isoformat(),
            'listing_date': getattr(listing, 'listing_date', None).isoformat() if getattr(listing, 'listing_date', None) else None,
            'metadata': safe_json_dumps(getattr(listing, 'metadata', {})),
            'raw_data': safe_json_dumps(getattr(listing, 'raw_data', {})),
            'is_validated': listing.is_validated,
            'validation_errors': safe_json_dumps(listing.validation_errors)
        }
    
    # === ENHANCED QUERY OPERATIONS ===
    
    def query_listings_advanced(self, 
                               query: str, 
                               params: List[Any] = None, 
                               limit: int = 100, 
                               offset: int = 0) -> List[Dict[str, Any]]:
        """
        Execute advanced custom queries with performance tracking
        
        Args:
            query: SQL query string
            params: Query parameters
            limit: Maximum number of results
            offset: Results offset
            
        Returns:
            List of result dictionaries
        """
        start_time = time.time()
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Add limit and offset if not present
                if 'LIMIT' not in query.upper():
                    query += f" LIMIT {limit} OFFSET {offset}"
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = [dict(row) for row in cursor.fetchall()]
                
                # Update performance metrics
                self._performance_metrics['queries_executed'] += 1
                self._performance_metrics['total_query_time'] += time.time() - start_time
                
                return results
                
        except Exception as e:
            logger.error(f"Advanced query failed: {e}")
            return []
    
    # === ENHANCED STATISTICS ===
    
    def get_enhanced_database_stats(self) -> DatabaseStats:
        """Get comprehensive database statistics with performance metrics"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Basic counts with single query
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM jewelry_listings) as total_listings,
                        (SELECT COUNT(*) FROM jewelry_images) as total_images,
                        (SELECT COUNT(*) FROM jewelry_specifications) as total_specifications,
                        (SELECT COUNT(*) FROM scraping_sessions) as total_sessions,
                        (SELECT AVG(data_completeness_score) FROM jewelry_listings) as avg_quality_score
                """)
                
                basic_stats = cursor.fetchone()
                
                # Category breakdown
                cursor.execute("SELECT category, COUNT(*) FROM jewelry_listings GROUP BY category")
                categories_breakdown = dict(cursor.fetchall())
                
                # Materials breakdown
                cursor.execute("SELECT material, COUNT(*) FROM jewelry_listings GROUP BY material")
                materials_breakdown = dict(cursor.fetchall())
                
                # Price range
                cursor.execute("SELECT MIN(price), MAX(price) FROM jewelry_listings WHERE price > 0")
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
                storage_size_mb = self.db_path.stat().st_size / (1024 * 1024)
                
                # Connection pool stats
                connection_pool_stats = self.connection_pool.get_stats()
                
                # Performance metrics
                performance_metrics = self._performance_metrics.copy()
                if performance_metrics['queries_executed'] > 0:
                    performance_metrics['avg_query_time'] = performance_metrics['total_query_time'] / performance_metrics['queries_executed']
                else:
                    performance_metrics['avg_query_time'] = 0.0
                
                return DatabaseStats(
                    total_listings=basic_stats[0],
                    total_images=basic_stats[1],
                    total_specifications=basic_stats[2],
                    total_sessions=basic_stats[3],
                    avg_quality_score=basic_stats[4] or 0.0,
                    categories_breakdown=categories_breakdown,
                    materials_breakdown=materials_breakdown,
                    price_range=price_range,
                    recent_activity=recent_activity,
                    storage_size_mb=storage_size_mb,
                    connection_pool_stats=connection_pool_stats,
                    performance_metrics=performance_metrics
                )
                
        except Exception as e:
            logger.error(f"Failed to get enhanced database stats: {e}")
            return DatabaseStats(0, 0, 0, 0, 0.0, {}, {}, (0.0, 0.0), {}, 0.0, {}, {})
    
    # === DATABASE OPTIMIZATION ===
    
    def optimize_database(self, full_optimization: bool = False) -> bool:
        """
        Enhanced database optimization with incremental and full modes
        
        Args:
            full_optimization: Perform full optimization including VACUUM
            
        Returns:
            bool: Success status
        """
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                logger.info("Starting database optimization...")
                
                # Update table statistics
                cursor.execute("ANALYZE")
                
                # Incremental vacuum for WAL mode
                if not full_optimization:
                    cursor.execute("PRAGMA incremental_vacuum")
                else:
                    # Full vacuum - WARNING: This can be slow and locks database
                    logger.info("Performing full database VACUUM (this may take a while)")
                    cursor.execute("VACUUM")
                
                # Optimize query planner
                cursor.execute("PRAGMA optimize")
                
                # Update checkpoint for WAL mode
                if self.enable_wal:
                    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                
                conn.commit()
                
                self._performance_metrics['last_optimization'] = datetime.now()
                logger.info("Database optimization completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    # === HEALTH MONITORING ===
    
    def check_database_health(self) -> Dict[str, Any]:
        """
        Comprehensive database health check
        
        Returns:
            Dict with health status and recommendations
        """
        health_report = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check database integrity
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                health_report['metrics']['integrity_check'] = integrity_result
                
                if integrity_result != "ok":
                    health_report['status'] = 'critical'
                    health_report['issues'].append(f"Database integrity check failed: {integrity_result}")
                
                # Check WAL mode status
                cursor.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]
                health_report['metrics']['journal_mode'] = journal_mode
                
                if journal_mode != 'wal' and self.enable_wal:
                    health_report['warnings'].append("WAL mode not enabled despite configuration")
                
                # Check database size and fragmentation
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]
                
                fragmentation_ratio = (freelist_count / page_count) * 100 if page_count > 0 else 0
                health_report['metrics']['fragmentation_percent'] = fragmentation_ratio
                
                if fragmentation_ratio > 25:
                    health_report['warnings'].append(f"High fragmentation detected: {fragmentation_ratio:.1f}%")
                    health_report['recommendations'].append("Consider running VACUUM to defragment database")
                
                # Check index usage
                cursor.execute("PRAGMA stats")
                health_report['metrics']['table_stats'] = cursor.fetchall()
                
                # Connection pool health
                pool_stats = self.connection_pool.get_stats()
                health_report['metrics']['connection_pool'] = pool_stats
                
                if pool_stats['active_connections'] / pool_stats['max_connections'] > 0.8:
                    health_report['warnings'].append("Connection pool utilization is high")
                    health_report['recommendations'].append("Consider increasing connection pool size")
                
                # Performance metrics
                perf_metrics = self._performance_metrics.copy()
                if perf_metrics['queries_executed'] > 0:
                    avg_query_time = perf_metrics['total_query_time'] / perf_metrics['queries_executed']
                    health_report['metrics']['avg_query_time_ms'] = avg_query_time * 1000
                    
                    if avg_query_time > 1.0:  # Queries taking more than 1 second on average
                        health_report['warnings'].append("Average query time is high")
                        health_report['recommendations'].append("Consider optimizing slow queries or adding indexes")
                
        except Exception as e:
            health_report['status'] = 'error'
            health_report['issues'].append(f"Health check failed: {e}")
        
        return health_report
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            if hasattr(self, 'connection_pool'):
                self.connection_pool._cleanup_pool()
        except:
            pass