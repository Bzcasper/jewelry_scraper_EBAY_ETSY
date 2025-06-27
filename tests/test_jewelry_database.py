#!/usr/bin/env python3
"""
Comprehensive Test Suite for Jewelry Database System
Tests all data operations, validation, export, analytics, and backup functionality.
"""

import os
import tempfile
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ..data.jewelry_data_manager import JewelryDatabaseManager, QueryFilters
from ..models.jewelry_models import (
    JewelryListing, JewelryImage, JewelrySpecification, ScrapingSession,
    JewelryCategory, JewelryMaterial, ListingStatus, ImageType
)


class JewelryDatabaseTester:
    """Comprehensive test suite for jewelry database operations"""

    def __init__(self):
        self.test_db_path = None
        self.db_manager = None
        self.temp_dir = None
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }

    def setup_test_environment(self):
        """Set up test environment with temporary database"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_jewelry.db")
        self.db_manager = JewelryDatabaseManager(self.test_db_path)
        print(f"🧪 Test database created: {self.test_db_path}")

    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Test environment cleaned up")

    def assert_test(self, condition: bool, test_name: str, error_msg: str = ""):
        """Assert test condition and track results"""
        if condition:
            print(f"✅ {test_name}")
            self.test_results['passed'] += 1
        else:
            print(f"❌ {test_name}: {error_msg}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"{test_name}: {error_msg}")

    def create_sample_listing(self, suffix: str = "") -> JewelryListing:
        """Create a sample jewelry listing for testing"""
        listing_id = f"test_listing_{uuid.uuid4().hex[:8]}{suffix}"

        return JewelryListing(
            id=listing_id,
            title=f"Test Diamond Ring {suffix}",
            price=1250.99,
            currency="USD",
            condition="New",
            seller_id=f"test_seller_{suffix}",
            listing_url=f"https://ebay.com/itm/{listing_id}",
            category=JewelryCategory.RINGS,
            material=JewelryMaterial.GOLD,
            brand="Test Brand",
            gemstone="Diamond",
            size="7",
            weight="5.2g",
            description="Beautiful diamond ring for testing purposes",
            features=["18K Gold", "Natural Diamond", "Certificate Included"],
            tags=["diamond", "engagement", "luxury"],
            image_count=5,
            seller_rating=98.5,
            seller_feedback_count=1523
        )

    def create_sample_image(self, listing_id: str) -> JewelryImage:
        """Create a sample jewelry image for testing"""
        return JewelryImage(
            image_id=f"img_{uuid.uuid4().hex[:8]}",
            listing_id=listing_id,
            original_url="https://example.com/image.jpg",
            local_path=f"/images/rings/{listing_id}_main.jpg",
            filename=f"{listing_id}_main.jpg",
            image_type=ImageType.MAIN,
            sequence_order=0,
            file_size=156789,
            width=800,
            height=600,
            format="jpg",
            is_processed=True,
            quality_score=0.95
        )

    def test_database_initialization(self):
        """Test database initialization"""
        print("\n🔧 Testing Database Initialization")
        print("-" * 40)

        # Test database initialization
        success = self.db_manager.initialize_database()
        self.assert_test(success, "Database initialization",
                         "Failed to initialize database")

        # Test database file exists
        self.assert_test(
            os.path.exists(self.test_db_path),
            "Database file creation",
            "Database file was not created"
        )

        # Test table creation
        table_sizes = self.db_manager.get_table_sizes()
        expected_tables = ['jewelry_listings', 'jewelry_images',
                           'jewelry_specifications', 'scraping_sessions']

        for table in expected_tables:
            self.assert_test(
                table in table_sizes,
                f"Table creation: {table}",
                f"Table {table} was not created"
            )

    def test_listing_operations(self):
        """Test listing CRUD operations"""
        print("\n📦 Testing Listing Operations")
        print("-" * 40)

        # Test single listing insertion
        sample_listing = self.create_sample_listing("001")
        success = self.db_manager.insert_listing(sample_listing)
        self.assert_test(success, "Single listing insertion",
                         "Failed to insert listing")

        # Test listing retrieval
        retrieved_listing = self.db_manager.get_listing(sample_listing.id)
        self.assert_test(
            retrieved_listing is not None,
            "Listing retrieval",
            "Failed to retrieve inserted listing"
        )

        if retrieved_listing:
            self.assert_test(
                retrieved_listing.title == sample_listing.title,
                "Listing data integrity",
                "Retrieved listing data doesn't match inserted data"
            )

        # Test batch insertion
        batch_listings = [self.create_sample_listing(
            f"{i:03d}") for i in range(2, 12)]
        inserted_count = self.db_manager.batch_insert_listings(batch_listings)
        self.assert_test(
            inserted_count == 10,
            "Batch listing insertion",
            f"Expected 10 insertions, got {inserted_count}"
        )

        # Test listing count
        stats = self.db_manager.get_database_stats()
        self.assert_test(
            stats.total_listings == 11,  # 1 + 10
            "Total listing count",
            f"Expected 11 listings, got {stats.total_listings}"
        )

    def test_query_operations(self):
        """Test query and filtering operations"""
        print("\n🔍 Testing Query Operations")
        print("-" * 40)

        # Test basic query
        filters = QueryFilters()
        all_listings = self.db_manager.query_listings(filters, limit=20)
        self.assert_test(
            len(all_listings) > 0,
            "Basic query operation",
            "No listings returned from basic query"
        )

        # Test category filter
        filters = QueryFilters(category=JewelryCategory.RINGS.value)
        ring_listings = self.db_manager.query_listings(filters)
        self.assert_test(
            all(listing.category == JewelryCategory.RINGS for listing in ring_listings),
            "Category filtering",
            "Category filter returned wrong results"
        )

        # Test price range filter
        filters = QueryFilters(min_price=1000.0, max_price=2000.0)
        price_filtered = self.db_manager.query_listings(filters)
        self.assert_test(
            all(1000.0 <= listing.price <= 2000.0 for listing in price_filtered),
            "Price range filtering",
            "Price filter returned out-of-range results"
        )

        # Test search functionality
        filters = QueryFilters(search_text="diamond")
        search_results = self.db_manager.query_listings(filters)
        self.assert_test(
            len(search_results) > 0,
            "Text search functionality",
            "Search returned no results for 'diamond'"
        )

        # Test quality score filter
        filters = QueryFilters(min_quality_score=0.5)
        quality_filtered = self.db_manager.query_listings(filters)
        self.assert_test(
            all(listing.data_quality_score >=
                0.5 for listing in quality_filtered),
            "Quality score filtering",
            "Quality filter returned low-quality results"
        )

    def test_image_operations(self):
        """Test image metadata operations"""
        print("\n🖼️  Testing Image Operations")
        print("-" * 40)

        # Get a sample listing
        filters = QueryFilters()
        listings = self.db_manager.query_listings(filters, limit=1)

        if not listings:
            print("⚠️  Skipping image tests - no listings available")
            return

        sample_listing = listings[0]

        # Test image insertion
        sample_image = self.create_sample_image(sample_listing.id)
        success = self.db_manager.insert_image(sample_image)
        self.assert_test(success, "Image insertion", "Failed to insert image")

        # Test image retrieval
        listing_images = self.db_manager.get_listing_images(sample_listing.id)
        self.assert_test(
            len(listing_images) > 0,
            "Image retrieval",
            "No images retrieved for listing"
        )

        if listing_images:
            retrieved_image = listing_images[0]
            self.assert_test(
                retrieved_image.listing_id == sample_listing.id,
                "Image-listing association",
                "Image not properly associated with listing"
            )

    def test_statistics_and_analytics(self):
        """Test statistics and analytics functionality"""
        print("\n📊 Testing Statistics and Analytics")
        print("-" * 40)

        # Test database statistics
        stats = self.db_manager.get_database_stats()

        self.assert_test(
            stats.total_listings > 0,
            "Statistics generation",
            "No statistics generated"
        )

        self.assert_test(
            len(stats.categories_breakdown) > 0,
            "Category breakdown",
            "No category breakdown generated"
        )

        self.assert_test(
            len(stats.materials_breakdown) > 0,
            "Materials breakdown",
            "No materials breakdown generated"
        )

        self.assert_test(
            stats.avg_quality_score >= 0,
            "Average quality score calculation",
            "Invalid average quality score"
        )

        self.assert_test(
            stats.storage_size_mb > 0,
            "Storage size calculation",
            "Invalid storage size calculation"
        )

    def test_export_functionality(self):
        """Test data export functionality"""
        print("\n📤 Testing Export Functionality")
        print("-" * 40)

        # Test CSV export
        csv_path = os.path.join(self.temp_dir, "test_export.csv")
        success = self.db_manager.export_to_csv(csv_path)
        self.assert_test(success, "CSV export", "Failed to export to CSV")

        if success:
            self.assert_test(
                os.path.exists(csv_path),
                "CSV file creation",
                "CSV file was not created"
            )

        # Test JSON export
        json_path = os.path.join(self.temp_dir, "test_export.json")
        success = self.db_manager.export_to_json(json_path)
        self.assert_test(success, "JSON export", "Failed to export to JSON")

        if success:
            self.assert_test(
                os.path.exists(json_path),
                "JSON file creation",
                "JSON file was not created"
            )

            # Test JSON content validity
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.assert_test(
                    isinstance(data, list) and len(data) > 0,
                    "JSON content validity",
                    "JSON export contains invalid data"
                )
            except json.JSONDecodeError:
                self.assert_test(False, "JSON content validity",
                                 "JSON export is not valid JSON")

    def test_data_validation(self):
        """Test data validation functionality"""
        print("\n✅ Testing Data Validation")
        print("-" * 40)

        # Test database integrity validation
        validation = self.db_manager.validate_database_integrity()

        self.assert_test(
            validation['integrity_check'],
            "Database integrity check",
            "Database integrity check failed"
        )

        self.assert_test(
            validation['foreign_key_check'],
            "Foreign key constraint check",
            "Foreign key constraints validation failed"
        )

        self.assert_test(
            validation['orphaned_images'] == 0,
            "Orphaned images check",
            f"Found {validation['orphaned_images']} orphaned images"
        )

        self.assert_test(
            validation['duplicate_listings'] == 0,
            "Duplicate listings check",
            f"Found {validation['duplicate_listings']} duplicate listings"
        )

    def test_backup_and_recovery(self):
        """Test backup and recovery functionality"""
        print("\n💾 Testing Backup and Recovery")
        print("-" * 40)

        # Test backup creation
        backup_path = os.path.join(self.temp_dir, "test_backup.db")

        try:
            created_backup = self.db_manager.create_backup(backup_path)
            self.assert_test(
                created_backup == backup_path,
                "Backup creation",
                "Failed to create backup"
            )

            self.assert_test(
                os.path.exists(backup_path),
                "Backup file existence",
                "Backup file was not created"
            )

            # Get original stats for comparison
            original_stats = self.db_manager.get_database_stats()

            # Test backup restore
            success = self.db_manager.restore_backup(backup_path)
            self.assert_test(success, "Backup restoration",
                             "Failed to restore from backup")

            # Verify restored data
            restored_stats = self.db_manager.get_database_stats()
            self.assert_test(
                restored_stats.total_listings == original_stats.total_listings,
                "Backup data integrity",
                "Restored data doesn't match original"
            )

        except Exception as e:
            self.assert_test(False, "Backup and recovery", str(e))

    def test_cleanup_operations(self):
        """Test data cleanup operations"""
        print("\n🧹 Testing Cleanup Operations")
        print("-" * 40)

        # Test dry run cleanup
        cleanup_results = self.db_manager.cleanup_old_data(
            days_old=365, dry_run=True)

        self.assert_test(
            'listings_to_delete' in cleanup_results,
            "Cleanup dry run",
            "Cleanup dry run didn't return expected results"
        )

        self.assert_test(
            cleanup_results['dry_run'] == True,
            "Dry run flag",
            "Dry run flag not set correctly"
        )

        # For fresh test data, we shouldn't have anything old enough to clean
        self.assert_test(
            cleanup_results['listings_to_delete'] == 0,
            "Cleanup detection (no old data)",
            f"Unexpected old data found: {cleanup_results['listings_to_delete']} listings"
        )

    def test_performance_operations(self):
        """Test performance optimization operations"""
        print("\n⚡ Testing Performance Operations")
        print("-" * 40)

        # Test database optimization
        success = self.db_manager.optimize_database()
        self.assert_test(success, "Database optimization",
                         "Failed to optimize database")

        # Test table size queries
        table_sizes = self.db_manager.get_table_sizes()
        expected_tables = ['jewelry_listings', 'jewelry_images',
                           'jewelry_specifications', 'scraping_sessions']

        for table in expected_tables:
            self.assert_test(
                table in table_sizes and isinstance(table_sizes[table], int),
                f"Table size query: {table}",
                f"Failed to get size for table {table}"
            )

    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting Jewelry Database Test Suite")
        print("=" * 50)

        try:
            self.setup_test_environment()

            # Run all test categories
            self.test_database_initialization()
            self.test_listing_operations()
            self.test_query_operations()
            self.test_image_operations()
            self.test_statistics_and_analytics()
            self.test_export_functionality()
            self.test_data_validation()
            self.test_backup_and_recovery()
            self.test_cleanup_operations()
            self.test_performance_operations()

            # Print summary
            print("\n" + "=" * 50)
            print("📋 TEST SUMMARY")
            print("=" * 50)
            print(f"✅ Tests Passed: {self.test_results['passed']}")
            print(f"❌ Tests Failed: {self.test_results['failed']}")
            total_tests = self.test_results['passed'] + \
                self.test_results['failed']
            success_rate = (
                self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
            print(f"📊 Success Rate: {success_rate:.1f}%")

            if self.test_results['errors']:
                print(f"\n❌ ERRORS:")
                for error in self.test_results['errors']:
                    print(f"   - {error}")

            if self.test_results['failed'] == 0:
                print("\n🎉 ALL TESTS PASSED! Database system is working correctly.")
                return True
            else:
                print(
                    f"\n⚠️  {self.test_results['failed']} tests failed. Please review the errors above.")
                return False

        except Exception as e:
            print(f"\n💥 Test suite crashed: {e}")
            return False

        finally:
            self.cleanup_test_environment()


def main():
    """Run the test suite"""
    tester = JewelryDatabaseTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
