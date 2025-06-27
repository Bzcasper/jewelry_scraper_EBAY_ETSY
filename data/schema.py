"""
Enhanced Database Schema for Jewelry Scraping System
Comprehensive SQLite schema with optimized indexes and performance views.
"""

# Enhanced database table creation SQL schemas
JEWELRY_SCHEMA_SQL = {
    "listings": """
        CREATE TABLE IF NOT EXISTS jewelry_listings (
            -- Primary identifiers
            listing_id TEXT PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            ebay_item_id TEXT,
            
            -- Core listing information
            title TEXT NOT NULL,
            price REAL NOT NULL CHECK(price > 0),
            original_price REAL CHECK(original_price > 0),
            buy_it_now_price REAL CHECK(buy_it_now_price > 0),
            starting_bid REAL CHECK(starting_bid > 0),
            reserve_price REAL CHECK(reserve_price > 0),
            currency TEXT DEFAULT 'USD' CHECK(length(currency) = 3),
            condition TEXT NOT NULL,
            availability TEXT,
            quantity_available INTEGER CHECK(quantity_available >= 0),
            quantity_sold INTEGER CHECK(quantity_sold >= 0),
            
            -- Seller information
            seller_name TEXT NOT NULL,
            seller_rating REAL CHECK(seller_rating >= 0 AND seller_rating <= 100),
            seller_feedback_count INTEGER CHECK(seller_feedback_count >= 0),
            seller_feedback_percentage REAL CHECK(seller_feedback_percentage >= 0 AND seller_feedback_percentage <= 100),
            seller_location TEXT,
            seller_store_name TEXT,
            seller_level TEXT,
            
            -- Jewelry categorization
            category TEXT NOT NULL,
            subcategory TEXT,
            ebay_category TEXT,
            custom_categories TEXT, -- JSON array
            
            -- Jewelry attributes
            brand TEXT,
            manufacturer TEXT,
            designer TEXT,
            collection TEXT,
            style TEXT,
            era TEXT,
            country_origin TEXT,
            
            -- Primary material and composition
            material TEXT NOT NULL,
            materials TEXT, -- JSON array
            metal_purity TEXT,
            metal_stamp TEXT,
            plating TEXT,
            finish TEXT,
            
            -- Measurements and physical properties
            size TEXT,
            ring_size TEXT,
            length TEXT,
            width TEXT,
            thickness TEXT,
            weight TEXT,
            carat_weight REAL CHECK(carat_weight >= 0),
            dimensions TEXT,
            
            -- Gemstone information
            main_stone TEXT,
            main_stone_details TEXT, -- JSON object
            stone_color TEXT,
            stone_clarity TEXT,
            stone_cut TEXT,
            stone_shape TEXT,
            stone_carat TEXT,
            stone_certification TEXT,
            accent_stones TEXT, -- JSON array
            
            -- Content and description
            description TEXT,
            description_html TEXT,
            features TEXT, -- JSON array
            specifications TEXT, -- JSON object
            tags TEXT, -- JSON array
            keywords TEXT, -- JSON array
            
            -- eBay specific details
            item_number TEXT,
            listing_type TEXT,
            listing_format TEXT,
            listing_status TEXT DEFAULT 'unknown',
            listing_duration TEXT,
            
            -- Engagement metrics
            watchers INTEGER CHECK(watchers >= 0),
            views INTEGER CHECK(views >= 0),
            page_views INTEGER CHECK(page_views >= 0),
            bids INTEGER CHECK(bids >= 0),
            bid_count INTEGER CHECK(bid_count >= 0),
            max_bid REAL CHECK(max_bid >= 0),
            reserve_met BOOLEAN,
            time_left TEXT,
            
            -- Pricing details
            best_offer BOOLEAN,
            price_drop REAL,
            discount_percentage REAL CHECK(discount_percentage >= 0 AND discount_percentage <= 100),
            
            -- Shipping and logistics
            shipping_cost REAL CHECK(shipping_cost >= 0),
            ships_from TEXT,
            ships_to TEXT,
            shipping_methods TEXT, -- JSON array
            expedited_shipping BOOLEAN,
            international_shipping BOOLEAN,
            handling_time TEXT,
            return_policy TEXT,
            return_period TEXT,
            
            -- Media information
            image_count INTEGER DEFAULT 0 CHECK(image_count >= 0),
            has_video BOOLEAN DEFAULT FALSE,
            video_urls TEXT, -- JSON array
            main_image_url TEXT,
            main_image_path TEXT,
            
            -- Quality metrics
            description_length INTEGER DEFAULT 0 CHECK(description_length >= 0),
            title_quality_score REAL DEFAULT 0.0 CHECK(title_quality_score >= 0 AND title_quality_score <= 1),
            description_quality_score REAL DEFAULT 0.0 CHECK(description_quality_score >= 0 AND description_quality_score <= 1),
            image_quality_score REAL DEFAULT 0.0 CHECK(image_quality_score >= 0 AND image_quality_score <= 1),
            data_completeness_score REAL DEFAULT 0.0 CHECK(data_completeness_score >= 0 AND data_completeness_score <= 1),
            completeness_score REAL DEFAULT 0.0 CHECK(completeness_score >= 0 AND completeness_score <= 1),
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            listing_date TIMESTAMP,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            last_seen TIMESTAMP,
            
            -- Validation and processing
            is_validated BOOLEAN DEFAULT FALSE,
            validation_errors TEXT, -- JSON array
            validation_score REAL DEFAULT 0.0 CHECK(validation_score >= 0 AND validation_score <= 1),
            data_source TEXT DEFAULT 'ebay',
            scraper_version TEXT,
            
            -- Deduplication and hashing
            content_hash TEXT,
            listing_hash TEXT,
            
            -- Metadata and raw data
            metadata TEXT, -- JSON object
            raw_data TEXT, -- JSON object
            extraction_metadata TEXT, -- JSON object
            processing_flags TEXT -- JSON object
        )
    """,
    
    "images": """
        CREATE TABLE IF NOT EXISTS jewelry_images (
            -- Primary identifiers
            image_id TEXT PRIMARY KEY,
            listing_id TEXT NOT NULL,
            
            -- Image source and storage
            original_url TEXT NOT NULL,
            local_path TEXT,
            filename TEXT,
            
            -- Enhanced image classification
            image_type TEXT DEFAULT 'gallery',
            sequence_order INTEGER DEFAULT 0 CHECK(sequence_order >= 0),
            is_primary BOOLEAN DEFAULT FALSE,
            
            -- Image properties
            file_size INTEGER CHECK(file_size >= 0),
            width INTEGER CHECK(width >= 0),
            height INTEGER CHECK(height >= 0),
            format TEXT,
            color_mode TEXT,
            bit_depth INTEGER,
            
            -- Enhanced quality and processing
            is_processed BOOLEAN DEFAULT FALSE,
            is_optimized BOOLEAN DEFAULT FALSE,
            is_resized BOOLEAN DEFAULT FALSE,
            quality_score REAL CHECK(quality_score >= 0 AND quality_score <= 1),
            resolution_score REAL CHECK(resolution_score >= 0 AND resolution_score <= 1),
            sharpness_score REAL CHECK(sharpness_score >= 0 AND sharpness_score <= 1),
            brightness_score REAL CHECK(brightness_score >= 0 AND brightness_score <= 1),
            
            -- Enhanced content analysis
            contains_text BOOLEAN DEFAULT FALSE,
            contains_watermark BOOLEAN DEFAULT FALSE,
            contains_logo BOOLEAN DEFAULT FALSE,
            is_duplicate BOOLEAN DEFAULT FALSE,
            similarity_hash TEXT,
            color_histogram TEXT,
            dominant_colors TEXT, -- JSON array
            
            -- AI/ML analysis
            ai_tags TEXT, -- JSON array
            ai_confidence REAL CHECK(ai_confidence >= 0 AND ai_confidence <= 1),
            object_detection TEXT, -- JSON array
            
            -- Descriptions and text
            alt_text TEXT,
            generated_description TEXT,
            manual_description TEXT,
            
            -- Processing history
            processing_history TEXT, -- JSON array
            optimization_settings TEXT, -- JSON object
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            downloaded_at TIMESTAMP,
            processed_at TIMESTAMP,
            last_verified TIMESTAMP,
            
            -- Error handling
            download_attempts INTEGER DEFAULT 0 CHECK(download_attempts >= 0),
            download_errors TEXT, -- JSON array
            processing_errors TEXT, -- JSON array
            
            -- Metadata
            metadata TEXT, -- JSON object
            exif_data TEXT, -- JSON object
            
            FOREIGN KEY (listing_id) REFERENCES jewelry_listings (listing_id) ON DELETE CASCADE
        )
    """,
    
    "specifications": """
        CREATE TABLE IF NOT EXISTS jewelry_specifications (
            -- Primary identifiers
            spec_id TEXT PRIMARY KEY,
            listing_id TEXT NOT NULL,
            
            -- Specification details
            attribute_name TEXT NOT NULL,
            attribute_value TEXT NOT NULL,
            attribute_category TEXT,
            attribute_type TEXT,
            
            -- Enhanced extraction information
            source_section TEXT,
            extraction_method TEXT,
            confidence_score REAL DEFAULT 0.0 CHECK(confidence_score >= 0 AND confidence_score <= 1),
            
            -- Enhanced standardization
            standardized_name TEXT,
            standardized_value TEXT,
            standardized_category TEXT,
            unit TEXT,
            unit_standardized TEXT,
            
            -- Value parsing
            numeric_value REAL,
            boolean_value BOOLEAN,
            date_value TIMESTAMP,
            
            -- Quality and validation
            is_verified BOOLEAN DEFAULT FALSE,
            is_standardized BOOLEAN DEFAULT FALSE,
            validation_status TEXT,
            quality_score REAL DEFAULT 0.0 CHECK(quality_score >= 0 AND quality_score <= 1),
            
            -- Relationships
            related_specs TEXT, -- JSON array
            conflicts_with TEXT, -- JSON array
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP,
            
            -- Metadata
            metadata TEXT, -- JSON object
            
            FOREIGN KEY (listing_id) REFERENCES jewelry_listings (listing_id) ON DELETE CASCADE
        )
    """,
    
    "scraping_sessions": """
        CREATE TABLE IF NOT EXISTS scraping_sessions (
            -- Primary identifiers
            session_id TEXT PRIMARY KEY,
            session_name TEXT,
            parent_session_id TEXT,
            
            -- Enhanced session configuration
            search_query TEXT,
            search_filters TEXT, -- JSON object
            search_categories TEXT, -- JSON array
            max_pages INTEGER CHECK(max_pages >= 1),
            max_listings INTEGER CHECK(max_listings >= 1),
            target_quality_score REAL CHECK(target_quality_score >= 0 AND target_quality_score <= 1),
            
            -- Enhanced session status
            status TEXT DEFAULT 'initialized',
            progress_percentage REAL DEFAULT 0.0 CHECK(progress_percentage >= 0 AND progress_percentage <= 100),
            current_phase TEXT,
            
            -- Enhanced statistics
            listings_found INTEGER DEFAULT 0 CHECK(listings_found >= 0),
            listings_scraped INTEGER DEFAULT 0 CHECK(listings_scraped >= 0),
            listings_failed INTEGER DEFAULT 0 CHECK(listings_failed >= 0),
            listings_skipped INTEGER DEFAULT 0 CHECK(listings_skipped >= 0),
            listings_duplicate INTEGER DEFAULT 0 CHECK(listings_duplicate >= 0),
            
            -- Image statistics
            images_found INTEGER DEFAULT 0 CHECK(images_found >= 0),
            images_downloaded INTEGER DEFAULT 0 CHECK(images_downloaded >= 0),
            images_failed INTEGER DEFAULT 0 CHECK(images_failed >= 0),
            images_skipped INTEGER DEFAULT 0 CHECK(images_skipped >= 0),
            
            -- Performance metrics
            pages_processed INTEGER DEFAULT 0 CHECK(pages_processed >= 0),
            requests_made INTEGER DEFAULT 0 CHECK(requests_made >= 0),
            requests_successful INTEGER DEFAULT 0 CHECK(requests_successful >= 0),
            requests_failed INTEGER DEFAULT 0 CHECK(requests_failed >= 0),
            data_volume_mb REAL DEFAULT 0.0 CHECK(data_volume_mb >= 0),
            
            -- Quality metrics
            average_quality_score REAL DEFAULT 0.0 CHECK(average_quality_score >= 0 AND average_quality_score <= 1),
            high_quality_count INTEGER DEFAULT 0 CHECK(high_quality_count >= 0),
            low_quality_count INTEGER DEFAULT 0 CHECK(low_quality_count >= 0),
            
            -- Timing information
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            paused_at TIMESTAMP,
            resumed_at TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            estimated_completion TIMESTAMP,
            
            -- Enhanced error handling
            error_count INTEGER DEFAULT 0 CHECK(error_count >= 0),
            warning_count INTEGER DEFAULT 0 CHECK(warning_count >= 0),
            critical_error_count INTEGER DEFAULT 0 CHECK(critical_error_count >= 0),
            last_error TEXT,
            error_categories TEXT, -- JSON object
            retry_count INTEGER DEFAULT 0 CHECK(retry_count >= 0),
            
            -- Enhanced configuration
            user_agent TEXT,
            proxy_used TEXT,
            rate_limit_delay REAL DEFAULT 1.0 CHECK(rate_limit_delay >= 0),
            concurrent_requests INTEGER DEFAULT 1 CHECK(concurrent_requests >= 1),
            timeout_seconds INTEGER DEFAULT 30 CHECK(timeout_seconds >= 1),
            
            -- Enhanced output configuration
            export_formats TEXT, -- JSON array
            output_directory TEXT,
            backup_enabled BOOLEAN DEFAULT TRUE,
            compression_enabled BOOLEAN DEFAULT FALSE,
            
            -- Advanced metrics
            performance_metrics TEXT, -- JSON object
            resource_usage TEXT, -- JSON object
            
            -- Session metadata
            metadata TEXT, -- JSON object
            tags TEXT, -- JSON array
            notes TEXT,
            
            FOREIGN KEY (parent_session_id) REFERENCES scraping_sessions (session_id) ON DELETE SET NULL
        )
    """,
    
    "analytics_cache": """
        CREATE TABLE IF NOT EXISTS analytics_cache (
            cache_key TEXT PRIMARY KEY,
            cache_value TEXT NOT NULL, -- JSON object
            cache_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            hit_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "data_quality_log": """
        CREATE TABLE IF NOT EXISTS data_quality_log (
            log_id TEXT PRIMARY KEY,
            listing_id TEXT,
            session_id TEXT,
            quality_type TEXT NOT NULL, -- 'validation', 'completeness', 'accuracy'
            before_score REAL,
            after_score REAL,
            changes_made TEXT, -- JSON object
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (listing_id) REFERENCES jewelry_listings (listing_id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES scraping_sessions (session_id) ON DELETE CASCADE
        )
    """,
    
    "export_history": """
        CREATE TABLE IF NOT EXISTS export_history (
            export_id TEXT PRIMARY KEY,
            export_type TEXT NOT NULL, -- 'csv', 'json', 'excel', 'xml'
            file_path TEXT,
            record_count INTEGER DEFAULT 0,
            file_size_mb REAL DEFAULT 0.0,
            filters_applied TEXT, -- JSON object
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            status TEXT DEFAULT 'completed' -- 'pending', 'completed', 'failed'
        )
    """
}

# Enhanced index creation SQL for maximum performance
JEWELRY_INDEXES_SQL = [
    # === PRIMARY TABLE INDEXES ===
    
    # Listings table - core indexes
    "CREATE INDEX IF NOT EXISTS idx_listings_category ON jewelry_listings(category)",
    "CREATE INDEX IF NOT EXISTS idx_listings_material ON jewelry_listings(material)",
    "CREATE INDEX IF NOT EXISTS idx_listings_price ON jewelry_listings(price)",
    "CREATE INDEX IF NOT EXISTS idx_listings_brand ON jewelry_listings(brand)",
    "CREATE INDEX IF NOT EXISTS idx_listings_seller ON jewelry_listings(seller_name)",
    "CREATE INDEX IF NOT EXISTS idx_listings_scraped_at ON jewelry_listings(scraped_at)",
    "CREATE INDEX IF NOT EXISTS idx_listings_status ON jewelry_listings(listing_status)",
    "CREATE INDEX IF NOT EXISTS idx_listings_completeness ON jewelry_listings(data_completeness_score)",
    "CREATE INDEX IF NOT EXISTS idx_listings_validated ON jewelry_listings(is_validated)",
    "CREATE INDEX IF NOT EXISTS idx_listings_condition ON jewelry_listings(condition)",
    "CREATE INDEX IF NOT EXISTS idx_listings_currency ON jewelry_listings(currency)",
    
    # Listings table - composite indexes for common query patterns
    "CREATE INDEX IF NOT EXISTS idx_listings_category_price ON jewelry_listings(category, price)",
    "CREATE INDEX IF NOT EXISTS idx_listings_material_category ON jewelry_listings(material, category)",
    "CREATE INDEX IF NOT EXISTS idx_listings_seller_status ON jewelry_listings(seller_name, listing_status)",
    "CREATE INDEX IF NOT EXISTS idx_listings_quality_scraped ON jewelry_listings(data_completeness_score, scraped_at)",
    "CREATE INDEX IF NOT EXISTS idx_listings_brand_category ON jewelry_listings(brand, category)",
    "CREATE INDEX IF NOT EXISTS idx_listings_price_range ON jewelry_listings(price, currency)",
    "CREATE INDEX IF NOT EXISTS idx_listings_status_time ON jewelry_listings(listing_status, scraped_at)",
    "CREATE INDEX IF NOT EXISTS idx_listings_category_material_price ON jewelry_listings(category, material, price)",
    
    # Listings table - search and filtering indexes
    "CREATE INDEX IF NOT EXISTS idx_listings_gemstone ON jewelry_listings(main_stone)",
    "CREATE INDEX IF NOT EXISTS idx_listings_size ON jewelry_listings(size)",
    "CREATE INDEX IF NOT EXISTS idx_listings_era ON jewelry_listings(era)",
    "CREATE INDEX IF NOT EXISTS idx_listings_designer ON jewelry_listings(designer)",
    "CREATE INDEX IF NOT EXISTS idx_listings_collection ON jewelry_listings(collection)",
    "CREATE INDEX IF NOT EXISTS idx_listings_seller_rating ON jewelry_listings(seller_rating)",
    "CREATE INDEX IF NOT EXISTS idx_listings_watchers ON jewelry_listings(watchers)",
    "CREATE INDEX IF NOT EXISTS idx_listings_views ON jewelry_listings(views)",
    "CREATE INDEX IF NOT EXISTS idx_listings_bids ON jewelry_listings(bids)",
    
    # Listings table - date range indexes
    "CREATE INDEX IF NOT EXISTS idx_listings_created_at ON jewelry_listings(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_listings_updated_at ON jewelry_listings(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_listings_listing_date ON jewelry_listings(listing_date)",
    "CREATE INDEX IF NOT EXISTS idx_listings_end_time ON jewelry_listings(end_time)",
    
    # Listings table - quality and validation indexes
    "CREATE INDEX IF NOT EXISTS idx_listings_validation_score ON jewelry_listings(validation_score)",
    "CREATE INDEX IF NOT EXISTS idx_listings_image_quality ON jewelry_listings(image_quality_score)",
    "CREATE INDEX IF NOT EXISTS idx_listings_title_quality ON jewelry_listings(title_quality_score)",
    "CREATE INDEX IF NOT EXISTS idx_listings_description_quality ON jewelry_listings(description_quality_score)",
    
    # Listings table - hash indexes for deduplication
    "CREATE INDEX IF NOT EXISTS idx_listings_content_hash ON jewelry_listings(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_listings_listing_hash ON jewelry_listings(listing_hash)",
    
    # === IMAGES TABLE INDEXES ===
    
    # Images table - basic indexes
    "CREATE INDEX IF NOT EXISTS idx_images_listing_id ON jewelry_images(listing_id)",
    "CREATE INDEX IF NOT EXISTS idx_images_type ON jewelry_images(image_type)",
    "CREATE INDEX IF NOT EXISTS idx_images_processed ON jewelry_images(is_processed)",
    "CREATE INDEX IF NOT EXISTS idx_images_duplicate ON jewelry_images(is_duplicate)",
    "CREATE INDEX IF NOT EXISTS idx_images_hash ON jewelry_images(similarity_hash)",
    "CREATE INDEX IF NOT EXISTS idx_images_primary ON jewelry_images(is_primary)",
    "CREATE INDEX IF NOT EXISTS idx_images_sequence ON jewelry_images(sequence_order)",
    
    # Images table - composite indexes
    "CREATE INDEX IF NOT EXISTS idx_images_listing_type ON jewelry_images(listing_id, image_type)",
    "CREATE INDEX IF NOT EXISTS idx_images_processed_type ON jewelry_images(is_processed, image_type)",
    "CREATE INDEX IF NOT EXISTS idx_images_listing_sequence ON jewelry_images(listing_id, sequence_order)",
    "CREATE INDEX IF NOT EXISTS idx_images_quality_processed ON jewelry_images(quality_score, is_processed)",
    
    # Images table - quality and analysis indexes
    "CREATE INDEX IF NOT EXISTS idx_images_quality_score ON jewelry_images(quality_score)",
    "CREATE INDEX IF NOT EXISTS idx_images_resolution_score ON jewelry_images(resolution_score)",
    "CREATE INDEX IF NOT EXISTS idx_images_contains_text ON jewelry_images(contains_text)",
    "CREATE INDEX IF NOT EXISTS idx_images_contains_watermark ON jewelry_images(contains_watermark)",
    "CREATE INDEX IF NOT EXISTS idx_images_ai_confidence ON jewelry_images(ai_confidence)",
    
    # Images table - file and processing indexes
    "CREATE INDEX IF NOT EXISTS idx_images_file_size ON jewelry_images(file_size)",
    "CREATE INDEX IF NOT EXISTS idx_images_format ON jewelry_images(format)",
    "CREATE INDEX IF NOT EXISTS idx_images_downloaded_at ON jewelry_images(downloaded_at)",
    "CREATE INDEX IF NOT EXISTS idx_images_processed_at ON jewelry_images(processed_at)",
    
    # === SPECIFICATIONS TABLE INDEXES ===
    
    # Specifications table - basic indexes
    "CREATE INDEX IF NOT EXISTS idx_specs_listing_id ON jewelry_specifications(listing_id)",
    "CREATE INDEX IF NOT EXISTS idx_specs_name ON jewelry_specifications(attribute_name)",
    "CREATE INDEX IF NOT EXISTS idx_specs_category ON jewelry_specifications(attribute_category)",
    "CREATE INDEX IF NOT EXISTS idx_specs_confidence ON jewelry_specifications(confidence_score)",
    "CREATE INDEX IF NOT EXISTS idx_specs_verified ON jewelry_specifications(is_verified)",
    "CREATE INDEX IF NOT EXISTS idx_specs_standardized ON jewelry_specifications(is_standardized)",
    
    # Specifications table - composite indexes
    "CREATE INDEX IF NOT EXISTS idx_specs_listing_name ON jewelry_specifications(listing_id, attribute_name)",
    "CREATE INDEX IF NOT EXISTS idx_specs_category_name ON jewelry_specifications(attribute_category, attribute_name)",
    "CREATE INDEX IF NOT EXISTS idx_specs_confidence_verified ON jewelry_specifications(confidence_score, is_verified)",
    
    # Specifications table - value type indexes
    "CREATE INDEX IF NOT EXISTS idx_specs_numeric_value ON jewelry_specifications(numeric_value)",
    "CREATE INDEX IF NOT EXISTS idx_specs_boolean_value ON jewelry_specifications(boolean_value)",
    "CREATE INDEX IF NOT EXISTS idx_specs_date_value ON jewelry_specifications(date_value)",
    "CREATE INDEX IF NOT EXISTS idx_specs_quality_score ON jewelry_specifications(quality_score)",
    
    # === SCRAPING SESSIONS TABLE INDEXES ===
    
    # Sessions table - basic indexes
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON scraping_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON scraping_sessions(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_query ON scraping_sessions(search_query)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_completed_at ON scraping_sessions(completed_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_progress ON scraping_sessions(progress_percentage)",
    
    # Sessions table - composite indexes
    "CREATE INDEX IF NOT EXISTS idx_sessions_status_started ON scraping_sessions(status, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_query_status ON scraping_sessions(search_query, status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_parent_status ON scraping_sessions(parent_session_id, status)",
    
    # Sessions table - performance indexes
    "CREATE INDEX IF NOT EXISTS idx_sessions_listings_scraped ON scraping_sessions(listings_scraped)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_quality_score ON scraping_sessions(average_quality_score)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_error_count ON scraping_sessions(error_count)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON scraping_sessions(last_activity)",
    
    # === ANALYTICS AND LOGGING INDEXES ===
    
    # Analytics cache indexes
    "CREATE INDEX IF NOT EXISTS idx_analytics_cache_type ON analytics_cache(cache_type)",
    "CREATE INDEX IF NOT EXISTS idx_analytics_cache_expires ON analytics_cache(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_analytics_cache_accessed ON analytics_cache(last_accessed)",
    "CREATE INDEX IF NOT EXISTS idx_analytics_cache_hits ON analytics_cache(hit_count)",
    
    # Data quality log indexes
    "CREATE INDEX IF NOT EXISTS idx_quality_log_listing ON data_quality_log(listing_id)",
    "CREATE INDEX IF NOT EXISTS idx_quality_log_session ON data_quality_log(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_quality_log_type ON data_quality_log(quality_type)",
    "CREATE INDEX IF NOT EXISTS idx_quality_log_created ON data_quality_log(created_at)",
    
    # Export history indexes
    "CREATE INDEX IF NOT EXISTS idx_export_type ON export_history(export_type)",
    "CREATE INDEX IF NOT EXISTS idx_export_created ON export_history(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_export_status ON export_history(status)",
    "CREATE INDEX IF NOT EXISTS idx_export_created_by ON export_history(created_by)"
]

# Enhanced database views for common queries and analytics
JEWELRY_VIEWS_SQL = [
    # High-quality listings view
    """
    CREATE VIEW IF NOT EXISTS high_quality_listings AS
    SELECT 
        listing_id,
        title,
        price,
        currency,
        category,
        material,
        brand,
        seller_name,
        seller_rating,
        data_completeness_score,
        image_count,
        watchers,
        views,
        scraped_at,
        url
    FROM jewelry_listings
    WHERE data_completeness_score >= 0.7 
      AND is_validated = TRUE
      AND image_count > 0
    """,
    
    # Listing summaries with enhanced metrics
    """
    CREATE VIEW IF NOT EXISTS listing_summaries AS
    SELECT 
        l.listing_id,
        l.title,
        l.price,
        l.original_price,
        l.currency,
        l.category,
        l.material,
        l.brand,
        l.seller_name,
        l.seller_rating,
        l.condition,
        l.data_completeness_score,
        l.image_count,
        l.watchers,
        l.views,
        l.bids,
        l.scraped_at,
        l.is_validated,
        l.url,
        COALESCE(img_stats.avg_quality, 0) as avg_image_quality,
        COALESCE(spec_stats.spec_count, 0) as specification_count
    FROM jewelry_listings l
    LEFT JOIN (
        SELECT 
            listing_id,
            AVG(quality_score) as avg_quality,
            COUNT(*) as image_count
        FROM jewelry_images 
        WHERE quality_score IS NOT NULL
        GROUP BY listing_id
    ) img_stats ON l.listing_id = img_stats.listing_id
    LEFT JOIN (
        SELECT 
            listing_id,
            COUNT(*) as spec_count
        FROM jewelry_specifications
        GROUP BY listing_id
    ) spec_stats ON l.listing_id = spec_stats.listing_id
    WHERE l.is_validated = TRUE
    """,
    
    # Enhanced category statistics
    """
    CREATE VIEW IF NOT EXISTS enhanced_category_stats AS
    SELECT 
        category,
        COUNT(*) as listing_count,
        AVG(price) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
        AVG(data_completeness_score) as avg_quality,
        AVG(image_count) as avg_images,
        AVG(seller_rating) as avg_seller_rating,
        COUNT(CASE WHEN watchers > 0 THEN 1 END) as listings_with_watchers,
        COUNT(CASE WHEN bids > 0 THEN 1 END) as listings_with_bids,
        AVG(watchers) as avg_watchers,
        AVG(views) as avg_views,
        COUNT(CASE WHEN listing_status = 'sold' THEN 1 END) as sold_count
    FROM jewelry_listings
    WHERE price > 0 AND is_validated = TRUE
    GROUP BY category
    HAVING COUNT(*) >= 5
    ORDER BY listing_count DESC
    """,
    
    # Enhanced seller statistics
    """
    CREATE VIEW IF NOT EXISTS enhanced_seller_stats AS
    SELECT 
        seller_name,
        COUNT(*) as listing_count,
        AVG(price) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(seller_rating) as avg_rating,
        AVG(seller_feedback_count) as avg_feedback_count,
        AVG(data_completeness_score) as avg_quality,
        AVG(image_count) as avg_images,
        COUNT(CASE WHEN watchers > 0 THEN 1 END) as listings_with_watchers,
        COUNT(CASE WHEN bids > 0 THEN 1 END) as listings_with_bids,
        COUNT(CASE WHEN listing_status = 'sold' THEN 1 END) as sold_count,
        AVG(watchers) as avg_watchers,
        AVG(views) as avg_views,
        COUNT(DISTINCT category) as categories_sold,
        MIN(scraped_at) as first_listing,
        MAX(scraped_at) as latest_listing
    FROM jewelry_listings
    WHERE seller_name IS NOT NULL AND is_validated = TRUE
    GROUP BY seller_name
    HAVING COUNT(*) >= 3
    ORDER BY listing_count DESC
    """,
    
    # Material and price analysis
    """
    CREATE VIEW IF NOT EXISTS material_price_analysis AS
    SELECT 
        material,
        category,
        COUNT(*) as listing_count,
        AVG(price) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as q1_price,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as q3_price,
        AVG(data_completeness_score) as avg_quality,
        COUNT(CASE WHEN main_stone IS NOT NULL AND main_stone != '' THEN 1 END) as with_gemstone_count
    FROM jewelry_listings
    WHERE price > 0 AND is_validated = TRUE
    GROUP BY material, category
    HAVING COUNT(*) >= 3
    ORDER BY material, category
    """,
    
    # Session performance summary
    """
    CREATE VIEW IF NOT EXISTS session_performance_summary AS
    SELECT 
        session_id,
        session_name,
        status,
        search_query,
        started_at,
        completed_at,
        CASE 
            WHEN completed_at IS NOT NULL THEN 
                ROUND((JULIANDAY(completed_at) - JULIANDAY(started_at)) * 24 * 60, 2)
            ELSE 
                ROUND((JULIANDAY('now') - JULIANDAY(started_at)) * 24 * 60, 2)
        END as duration_minutes,
        progress_percentage,
        listings_found,
        listings_scraped,
        listings_failed,
        CASE 
            WHEN (listings_scraped + listings_failed) > 0 THEN
                ROUND((listings_scraped * 100.0) / (listings_scraped + listings_failed), 2)
            ELSE 0
        END as success_rate_percent,
        images_downloaded,
        images_failed,
        CASE 
            WHEN (images_downloaded + images_failed) > 0 THEN
                ROUND((images_downloaded * 100.0) / (images_downloaded + images_failed), 2)
            ELSE 0
        END as image_success_rate_percent,
        pages_processed,
        requests_made,
        CASE 
            WHEN requests_made > 0 THEN
                ROUND((requests_successful * 100.0) / requests_made, 2)
            ELSE 0
        END as request_success_rate_percent,
        data_volume_mb,
        error_count,
        average_quality_score
    FROM scraping_sessions
    ORDER BY started_at DESC
    """,
    
    # Daily scraping activity summary
    """
    CREATE VIEW IF NOT EXISTS daily_activity_summary AS
    SELECT 
        DATE(scraped_at) as scraping_date,
        COUNT(*) as listings_scraped,
        COUNT(DISTINCT seller_name) as unique_sellers,
        COUNT(DISTINCT category) as categories_covered,
        AVG(price) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(data_completeness_score) as avg_quality,
        SUM(image_count) as total_images,
        COUNT(CASE WHEN is_validated = TRUE THEN 1 END) as validated_listings,
        COUNT(CASE WHEN data_completeness_score >= 0.8 THEN 1 END) as high_quality_listings
    FROM jewelry_listings
    WHERE scraped_at >= DATE('now', '-30 days')
    GROUP BY DATE(scraped_at)
    ORDER BY scraping_date DESC
    """,
    
    # Image quality analysis
    """
    CREATE VIEW IF NOT EXISTS image_quality_analysis AS
    SELECT 
        l.category,
        l.material,
        COUNT(i.image_id) as total_images,
        AVG(i.quality_score) as avg_quality,
        AVG(i.resolution_score) as avg_resolution,
        AVG(i.sharpness_score) as avg_sharpness,
        AVG(i.brightness_score) as avg_brightness,
        COUNT(CASE WHEN i.is_processed = TRUE THEN 1 END) as processed_count,
        COUNT(CASE WHEN i.is_duplicate = TRUE THEN 1 END) as duplicate_count,
        COUNT(CASE WHEN i.contains_text = TRUE THEN 1 END) as with_text_count,
        COUNT(CASE WHEN i.contains_watermark = TRUE THEN 1 END) as with_watermark_count,
        AVG(i.file_size) / 1024.0 as avg_size_kb
    FROM jewelry_listings l
    INNER JOIN jewelry_images i ON l.listing_id = i.listing_id
    WHERE l.is_validated = TRUE
    GROUP BY l.category, l.material
    HAVING COUNT(i.image_id) >= 10
    ORDER BY l.category, l.material
    """
]

# Database optimization and maintenance SQL
MAINTENANCE_SQL = {
    "analyze_tables": "ANALYZE",
    "vacuum_database": "VACUUM",
    "incremental_vacuum": "PRAGMA incremental_vacuum",
    "optimize_database": "PRAGMA optimize",
    "integrity_check": "PRAGMA integrity_check",
    "foreign_key_check": "PRAGMA foreign_key_check",
    "rebuild_indexes": """
        DROP INDEX IF EXISTS idx_listings_category_price;
        CREATE INDEX idx_listings_category_price ON jewelry_listings(category, price);
    """
}

# Performance monitoring queries
PERFORMANCE_QUERIES = {
    "table_sizes": """
        SELECT 
            name as table_name,
            (SELECT COUNT(*) FROM main.sqlite_master WHERE type='table' AND name=outer.name) as exists,
            CASE name
                WHEN 'jewelry_listings' THEN (SELECT COUNT(*) FROM jewelry_listings)
                WHEN 'jewelry_images' THEN (SELECT COUNT(*) FROM jewelry_images)
                WHEN 'jewelry_specifications' THEN (SELECT COUNT(*) FROM jewelry_specifications)
                WHEN 'scraping_sessions' THEN (SELECT COUNT(*) FROM scraping_sessions)
                ELSE 0
            END as row_count
        FROM (
            SELECT 'jewelry_listings' as name UNION ALL
            SELECT 'jewelry_images' UNION ALL
            SELECT 'jewelry_specifications' UNION ALL
            SELECT 'scraping_sessions'
        ) outer
    """,
    
    "index_usage": """
        SELECT 
            name,
            sql
        FROM sqlite_master 
        WHERE type = 'index' 
        AND name LIKE 'idx_%'
        ORDER BY name
    """,
    
    "database_size": """
        SELECT 
            page_count * page_size / 1024.0 / 1024.0 as size_mb,
            page_count,
            page_size,
            freelist_count,
            ROUND((freelist_count * 100.0) / page_count, 2) as fragmentation_percent
        FROM (
            SELECT 
                (SELECT * FROM pragma_page_count()) as page_count,
                (SELECT * FROM pragma_page_size()) as page_size,
                (SELECT * FROM pragma_freelist_count()) as freelist_count
        )
    """
}