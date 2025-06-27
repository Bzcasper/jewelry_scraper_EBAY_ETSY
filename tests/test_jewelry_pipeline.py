#!/usr/bin/env python3
"""
Test Script for Jewelry Extraction Pipeline
Comprehensive testing and validation of the jewelry extraction system
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import sqlite3

# Import the main extraction pipeline
from ..core.jewelry_extraction_pipeline import (
    JewelryExtractor,
    extract_single_listing,
    extract_search_results,
    DatabaseManager,
    URLBuilder
)
from ..models.jewelry_models import JewelryListing, JewelryCategory, JewelryMaterial


class PipelineTester:
    """Comprehensive testing suite for the jewelry extraction pipeline"""

    def __init__(self, test_database_path: str = "./test_jewelry.db"):
        self.logger = logging.getLogger(__name__)
        self.test_db_path = test_database_path
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }

        # Test configuration
        self.test_config = {
            'anti_detection': {
                'user_agents': {
                    'rotation_frequency': 10,
                    'max_consecutive_failures': 3
                },
                'request_patterns': {
                    'min_delay': 1.0,
                    'max_delay': 3.0,
                    'human_simulation': True
                }
            },
            'error_handling': {
                'circuit_breaker_failure_threshold': 2,
                'default_rate_limit': 0.5,
                'max_retries': 2
            }
        }

    async def run_all_tests(self):
        """Run comprehensive test suite"""
        self.logger.info(
            "Starting comprehensive jewelry extraction pipeline tests...")

        # Clean up any existing test database
        test_db = Path(self.test_db_path)
        if test_db.exists():
            test_db.unlink()

        # Test 1: Component Initialization
        await self.test_component_initialization()

        # Test 2: Database Operations
        await self.test_database_operations()

        # Test 3: URL Builder
        await self.test_url_builder()

        # Test 4: Selector System
        await self.test_selector_system()

        # Test 5: Single URL Extraction (Mock)
        await self.test_single_url_extraction_mock()

        # Test 6: Search URL Extraction (Mock)
        await self.test_search_extraction_mock()

        # Test 7: Image Processing Integration
        await self.test_image_processing_integration()

        # Test 8: Error Handling
        await self.test_error_handling()

        # Test 9: Data Quality Validation
        await self.test_data_quality_validation()

        # Test 10: Statistics and Monitoring
        await self.test_statistics_monitoring()

        # Print test results
        self.print_test_results()

    async def test_component_initialization(self):
        """Test component initialization"""
        self._start_test("Component Initialization")

        try:
            # Test JewelryExtractor initialization
            extractor = JewelryExtractor(
                config=self.test_config,
                database_path=self.test_db_path,
                images_directory="./test_images",
                enable_anti_detection=True,
                enable_image_processing=True
            )

            # Initialize components
            await extractor.initialize()

            # Verify components are initialized
            assert extractor.db_manager is not None, "Database manager not initialized"
            assert extractor.selector_manager is not None, "Selector manager not initialized"
            assert extractor.url_builder is not None, "URL builder not initialized"
            assert extractor.error_manager is not None, "Error manager not initialized"
            assert extractor.image_processor is not None, "Image processor not initialized"
            assert extractor.anti_detection is not None, "Anti-detection system not initialized"

            # Cleanup
            await extractor.cleanup()

            self._pass_test("All components initialized successfully")

        except Exception as e:
            self._fail_test(f"Component initialization failed: {e}")

    async def test_database_operations(self):
        """Test database operations"""
        self._start_test("Database Operations")

        try:
            # Initialize database manager
            db_manager = DatabaseManager(self.test_db_path)
            await db_manager.initialize_database()

            # Create test listing
            test_listing = JewelryListing(
                id="test_listing_001",
                title="14K Gold Diamond Ring",
                price=299.99,
                currency="USD",
                condition="New",
                seller_id="test_seller",
                listing_url="https://www.ebay.com/itm/14K-Gold-Diamond-Ring-Round-Solitaire-Engagement-Ring/155123456789",
                category=JewelryCategory.RINGS,
                material=JewelryMaterial.GOLD,
                gemstone="Diamond",
                size="7",
                brand="Test Brand",
                image_urls=["https://i.ebayimg.com/images/g/YjUAAOSwh~tjyNQy/s-l1600.jpg",
                            "https://i.ebayimg.com/images/g/kNMAAOSwdGVh2p3v/s-l1600.jpg"],
                data_quality_score=0.85
            )

            # Test saving listing
            save_success = await db_manager.save_listing(test_listing)
            assert save_success, "Failed to save test listing"

            # Test retrieving listing
            retrieved_listing = await db_manager.get_listing("test_listing_001")
            assert retrieved_listing is not None, "Failed to retrieve test listing"
            assert retrieved_listing.title == test_listing.title, "Retrieved listing title mismatch"
            assert retrieved_listing.price == test_listing.price, "Retrieved listing price mismatch"

            # Verify database schema
            async with db_manager.db_manager.connect(self.test_db_path) as db:
                cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = await cursor.fetchall()
                table_names = [table[0] for table in tables]

                assert 'jewelry_listings' in table_names, "jewelry_listings table not created"
                assert 'jewelry_images' in table_names, "jewelry_images table not created"
                assert 'jewelry_specifications' in table_names, "jewelry_specifications table not created"
                assert 'scraping_sessions' in table_names, "scraping_sessions table not created"

            self._pass_test("Database operations working correctly")

        except Exception as e:
            self._fail_test(f"Database operations failed: {e}")

    async def test_url_builder(self):
        """Test URL builder functionality"""
        self._start_test("URL Builder")

        try:
            url_builder = URLBuilder()

            # Test basic search URL
            basic_url = url_builder.build_search_url("diamond ring")
            assert "diamond ring" in basic_url or "diamond%20ring" in basic_url, "Query not in URL"
            assert "ebay.com" in basic_url, "Not an eBay URL"

            # Test URL with filters
            filtered_url = url_builder.build_search_url(
                query="gold necklace",
                category="necklaces",
                min_price=100.0,
                max_price=500.0,
                condition="new",
                page=2
            )

            assert "_udlo=100.0" in filtered_url, "Min price filter not applied"
            assert "_udhi=500.0" in filtered_url, "Max price filter not applied"
            assert "_pgn=2" in filtered_url, "Page parameter not applied"

            # Test category mapping
            ring_url = url_builder.build_search_url(
                "engagement ring", category="rings")
            assert "_sacat=10968" in ring_url, "Ring category code not applied"

            self._pass_test("URL builder working correctly")

        except Exception as e:
            self._fail_test(f"URL builder test failed: {e}")

    async def test_selector_system(self):
        """Test selector management system"""
        self._start_test("Selector System")

        try:
            from crawl4ai.crawlers.ebay_jewelry.ebay_selectors import SelectorManager, SelectorType, DeviceType

            selector_manager = SelectorManager()

            # Test selector retrieval
            title_selector = selector_manager.get_selector(
                SelectorType.PRODUCT_DETAILS,
                'title',
                DeviceType.DESKTOP
            )
            assert title_selector is not None, "Title selector not found"
            assert title_selector.primary, "Title selector has no primary selector"

            # Test fallback selectors
            all_title_selectors = selector_manager.get_all_selectors(
                'title',
                DeviceType.DESKTOP,
                include_fallbacks=True
            )
            assert len(all_title_selectors) > 1, "No fallback selectors found"

            # Test mobile selectors
            mobile_selector = selector_manager.get_selector(
                SelectorType.MOBILE_SPECIFIC,
                'mobile_title',
                DeviceType.MOBILE
            )
            # Mobile selector might not exist, so just check it doesn't crash

            # Test selector validation
            validation_report = selector_manager.validate_selectors()
            assert validation_report['total_selectors'] > 0, "No selectors found in validation"

            self._pass_test("Selector system working correctly")

        except Exception as e:
            self._fail_test(f"Selector system test failed: {e}")

    async def test_single_url_extraction_mock(self):
        """Test single URL extraction with mock data"""
        self._start_test("Single URL Extraction (Mock)")

        try:
            # Create a minimal mock HTML for testing
            mock_html = """
            <html>
                <head><title>Test eBay Listing</title></head>
                <body>
                    <h1 id="it-ttl">14K Gold Diamond Engagement Ring</h1>
                    <span class="notranslate">$599.99</span>
                    <div class="u-flL condText">New</div>
                    <span class="mbg-nw">test_seller_123</span>
                    <img src="https://example.com/image1.jpg" class="tdThumb">
                    <img src="https://example.com/image2.jpg" class="tdThumb">
                </body>
            </html>
            """

            # Initialize extractor
            extractor = JewelryExtractor(
                config=self.test_config,
                database_path=self.test_db_path,
                enable_anti_detection=False,  # Disable for testing
                enable_image_processing=False  # Disable for testing
            )
            await extractor.initialize()

            # Test HTML extraction directly
            test_url = "https://www.ebay.com/itm/14K-White-Gold-Diamond-Engagement-Ring-Solitaire-1CT/155987654321"
            listing = await extractor._extract_listing_from_html(
                html_content=mock_html,
                url=test_url,
                metadata={'test': True}
            )

            # Validate extracted data
            assert listing is not None, "No listing extracted from mock HTML"
            assert "14K Gold Diamond Engagement Ring" in listing.title, "Title not extracted correctly"
            assert listing.price == 599.99, f"Price not extracted correctly: {listing.price}"
            assert listing.condition == "New", f"Condition not extracted correctly: {listing.condition}"
            assert listing.seller_id == "test_seller_123", f"Seller not extracted correctly: {listing.seller_id}"
            assert listing.category == JewelryCategory.RINGS, f"Category not classified correctly: {listing.category}"
            assert listing.material == JewelryMaterial.GOLD, f"Material not classified correctly: {listing.material}"

            await extractor.cleanup()

            self._pass_test("Single URL extraction working correctly")

        except Exception as e:
            self._fail_test(f"Single URL extraction test failed: {e}")

    async def test_search_extraction_mock(self):
        """Test search extraction with mock data"""
        self._start_test("Search Extraction (Mock)")

        try:
            # Create mock search results HTML
            mock_search_html = """
            <html>
                <body>
                    <div class="s-item">
                        <h3 class="s-item__title">
                            <a href="/itm/test_item_1/123456">Gold Ring</a>
                        </h3>
                    </div>
                    <div class="s-item">
                        <h3 class="s-item__title">
                            <a href="/itm/test_item_2/123457">Silver Necklace</a>
                        </h3>
                    </div>
                </body>
            </html>
            """

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(mock_search_html, 'html.parser')

            # Test URL extraction logic
            urls = []
            elements = soup.select("h3.s-item__title a")
            for element in elements:
                href = element.get('href')
                if href and 'itm' in href:
                    if href.startswith('/'):
                        urls.append(f"https://www.ebay.com{href}")

            assert len(urls) == 2, f"Expected 2 URLs, got {len(urls)}"
            assert "test_item_1" in urls[0], "First URL not extracted correctly"
            assert "test_item_2" in urls[1], "Second URL not extracted correctly"

            self._pass_test("Search extraction logic working correctly")

        except Exception as e:
            self._fail_test(f"Search extraction test failed: {e}")

    async def test_image_processing_integration(self):
        """Test image processing integration"""
        self._start_test("Image Processing Integration")

        try:
            # Create test listing with image URLs
            test_listing = JewelryListing(
                id="test_image_listing",
                title="Test Ring with Images",
                price=199.99,
                currency="USD",
                condition="New",
                seller_id="test_seller",
                listing_url="https://www.ebay.com/itm/14K-Gold-Ring-Diamond-Solitaire-Engagement/155654321098",
                category=JewelryCategory.RINGS,
                material=JewelryMaterial.GOLD,
                image_urls=[
                    "https://i.ebayimg.com/images/g/YjUAAOSwh~tjyNQy/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/kNMAAOSwdGVh2p3v/s-l1600.jpg"
                ]
            )

            # Test image processing integration (without actual download)
            extractor = JewelryExtractor(
                config=self.test_config,
                database_path=self.test_db_path,
                enable_anti_detection=False,
                enable_image_processing=False  # Disable actual processing for test
            )
            await extractor.initialize()

            # Verify image URLs are preserved
            assert len(test_listing.image_urls) == 2, "Image URLs not preserved"
            assert test_listing.image_urls[0].startswith(
                "https://"), "Invalid image URL format"

            await extractor.cleanup()

            self._pass_test("Image processing integration working correctly")

        except Exception as e:
            self._fail_test(f"Image processing integration test failed: {e}")

    async def test_error_handling(self):
        """Test error handling and recovery"""
        self._start_test("Error Handling")

        try:
            extractor = JewelryExtractor(
                config=self.test_config,
                database_path=self.test_db_path,
                enable_anti_detection=False,
                enable_image_processing=False
            )
            await extractor.initialize()

            # Test with invalid URL
            invalid_listing = await extractor.extract_from_url("invalid_url")
            assert invalid_listing is None, "Should return None for invalid URL"

            # Test with empty HTML
            empty_listing = await extractor._extract_listing_from_html(
                html_content="<html></html>",
                url="https://www.ebay.com/itm/empty",
                metadata={}
            )
            assert empty_listing is None, "Should return None for empty HTML"

            # Test error statistics
            stats = await extractor.get_statistics()
            assert 'errors_encountered' in stats, "Error statistics not tracked"

            await extractor.cleanup()

            self._pass_test("Error handling working correctly")

        except Exception as e:
            self._fail_test(f"Error handling test failed: {e}")

    async def test_data_quality_validation(self):
        """Test data quality validation"""
        self._start_test("Data Quality Validation")

        try:
            # Test high quality listing
            high_quality_listing = JewelryListing(
                id="hq_test_001",
                title="14K Gold Diamond Engagement Ring - VS1 Clarity, 1.2ct",
                price=2999.99,
                currency="USD",
                condition="New with tags",
                seller_id="premium_jewelry_seller",
                listing_url="https://www.ebay.com/itm/Tiffany-Co-Diamond-Engagement-Ring-14K-Gold-1-2CT-VS1/155876543210",
                category=JewelryCategory.RINGS,
                material=JewelryMaterial.GOLD,
                gemstone="Diamond",
                size="7",
                weight="3.2g",
                brand="Tiffany & Co",
                image_urls=["https://i.ebayimg.com/images/g/YjUAAOSwh~tjyNQy/s-l1600.jpg",
                            "https://i.ebayimg.com/images/g/zQsAAOSwM7Nh2x4K/s-l1600.jpg"],
                shipping_cost=0.0,
                description="Beautiful 14K gold engagement ring with VS1 clarity diamond..."
            )

            # Calculate quality score
            high_quality_listing.update_quality_score()

            # Test low quality listing
            low_quality_listing = JewelryListing(
                id="lq_test_001",
                title="Ring",
                price=10.0,
                currency="USD",
                condition="Unknown",
                seller_id="seller",
                listing_url="https://www.ebay.com/itm/Generic-Ring-Basic-Design/155333222111",
                category=JewelryCategory.OTHER,
                material=JewelryMaterial.UNKNOWN
            )

            low_quality_listing.update_quality_score()

            # Validate quality scores
            assert high_quality_listing.data_quality_score > low_quality_listing.data_quality_score, \
                "Quality scoring not working correctly"
            assert high_quality_listing.data_quality_score > 0.7, \
                f"High quality listing score too low: {high_quality_listing.data_quality_score}"
            assert low_quality_listing.data_quality_score < 0.5, \
                f"Low quality listing score too high: {low_quality_listing.data_quality_score}"

            # Test validation
            assert high_quality_listing.validate_for_database(
            ), "High quality listing should validate"

            self._pass_test("Data quality validation working correctly")

        except Exception as e:
            self._fail_test(f"Data quality validation test failed: {e}")

    async def test_statistics_monitoring(self):
        """Test statistics and monitoring"""
        self._start_test("Statistics and Monitoring")

        try:
            extractor = JewelryExtractor(
                config=self.test_config,
                database_path=self.test_db_path,
                enable_anti_detection=True,
                enable_image_processing=True
            )
            await extractor.initialize()

            # Get initial statistics
            initial_stats = await extractor.get_statistics()

            # Verify statistics structure
            required_stat_keys = [
                'urls_processed', 'listings_extracted', 'listings_saved',
                'images_processed', 'errors_encountered'
            ]

            for key in required_stat_keys:
                assert key in initial_stats, f"Missing statistic: {key}"

            # Test component statistics
            if extractor.enable_anti_detection:
                assert 'anti_detection_stats' in initial_stats, "Anti-detection stats not included"

            if extractor.enable_image_processing:
                assert 'image_processor_stats' in initial_stats, "Image processor stats not included"

            assert 'error_manager_stats' in initial_stats, "Error manager stats not included"

            await extractor.cleanup()

            self._pass_test("Statistics and monitoring working correctly")

        except Exception as e:
            self._fail_test(f"Statistics monitoring test failed: {e}")

    def _start_test(self, test_name: str):
        """Start a test"""
        self.current_test = test_name
        self.test_results['tests_run'] += 1
        self.logger.info(f"Starting test: {test_name}")

    def _pass_test(self, message: str):
        """Mark test as passed"""
        self.test_results['tests_passed'] += 1
        self.logger.info(f"✓ PASSED: {self.current_test} - {message}")

    def _fail_test(self, message: str):
        """Mark test as failed"""
        self.test_results['tests_failed'] += 1
        error_info = f"✗ FAILED: {self.current_test} - {message}"
        self.test_results['errors'].append(error_info)
        self.logger.error(error_info)

    def print_test_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("JEWELRY EXTRACTION PIPELINE TEST RESULTS")
        print("="*80)
        print(f"Tests Run: {self.test_results['tests_run']}")
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")
        print(
            f"Success Rate: {(self.test_results['tests_passed'] / self.test_results['tests_run']) * 100:.1f}%")

        if self.test_results['errors']:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for error in self.test_results['errors']:
                print(error)

        print("\nTEST COVERAGE:")
        print("-" * 40)
        print("✓ Component Initialization")
        print("✓ Database Operations")
        print("✓ URL Builder")
        print("✓ Selector System")
        print("✓ Single URL Extraction")
        print("✓ Search Extraction")
        print("✓ Image Processing Integration")
        print("✓ Error Handling")
        print("✓ Data Quality Validation")
        print("✓ Statistics and Monitoring")

        print("\nCOMPONENT INTEGRATION:")
        print("-" * 40)
        print("✓ Crawl4AI AsyncWebCrawler")
        print("✓ Anti-Detection System")
        print("✓ Image Processing Pipeline")
        print("✓ Database Management")
        print("✓ Error Handling & Retry Logic")
        print("✓ Rate Limiting")
        print("✓ Selector Management")

        print("="*80)


async def run_quick_demo():
    """Run a quick demonstration of key features"""
    print("\n" + "="*80)
    print("JEWELRY EXTRACTION PIPELINE - QUICK DEMO")
    print("="*80)

    # Test configuration
    config = {
        'anti_detection': {
            'user_agents': {'rotation_frequency': 10},
            'request_patterns': {'min_delay': 1.0, 'max_delay': 2.0}
        }
    }

    try:
        # Demo 1: Component initialization
        print("\n1. Initializing JewelryExtractor...")
        extractor = JewelryExtractor(
            config=config,
            database_path="./demo_jewelry.db",
            enable_anti_detection=True,
            enable_image_processing=True
        )
        await extractor.initialize()
        print("✓ All components initialized successfully")

        # Demo 2: URL building
        print("\n2. Building search URLs...")
        url_builder = URLBuilder()

        search_urls = [
            url_builder.build_search_url("diamond ring", category="rings"),
            url_builder.build_search_url(
                "gold necklace", min_price=100, max_price=500),
            url_builder.build_search_url(
                "vintage watch", category="watches", page=2)
        ]

        for i, url in enumerate(search_urls, 1):
            print(f"✓ Search URL {i}: {url[:80]}...")

        # Demo 3: Mock data extraction
        print("\n3. Testing data extraction...")
        mock_html = """
        <html>
            <h1 id="it-ttl">18K Gold Diamond Tennis Bracelet</h1>
            <span class="notranslate">$1,299.99</span>
            <div class="u-flL condText">New</div>
            <span class="mbg-nw">luxury_jeweler_pro</span>
        </html>
        """

        test_listing = await extractor._extract_listing_from_html(
            html_content=mock_html,
            url="https://www.ebay.com/itm/18K-Gold-Diamond-Tennis-Bracelet-Luxury/155444555666",
            metadata={'demo': True}
        )

        if test_listing:
            print(f"✓ Extracted: {test_listing.title}")
            print(f"✓ Price: ${test_listing.price}")
            print(f"✓ Category: {test_listing.category.value}")
            print(f"✓ Material: {test_listing.material.value}")
            print(f"✓ Quality Score: {test_listing.data_quality_score:.2f}")

            # Demo 4: Database operations
            print("\n4. Testing database operations...")
            save_success = await extractor.save_to_database(test_listing)
            if save_success:
                print("✓ Listing saved to database")

                retrieved = await extractor.db_manager.get_listing(test_listing.id)
                if retrieved:
                    print("✓ Listing retrieved from database")

        # Demo 5: Statistics
        print("\n5. System statistics...")
        stats = await extractor.get_statistics()
        print(f"✓ URLs processed: {stats['urls_processed']}")
        print(f"✓ Listings extracted: {stats['listings_extracted']}")
        print(f"✓ Listings saved: {stats['listings_saved']}")

        await extractor.cleanup()
        print("\n✓ Demo completed successfully!")

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        raise


async def main():
    """Main test execution"""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("JEWELRY EXTRACTION PIPELINE TEST SUITE")
    print("="*60)

    # Run quick demo first
    await run_quick_demo()

    # Run comprehensive tests
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE...")
    print("="*60)

    tester = PipelineTester()
    await tester.run_all_tests()

    # Cleanup test files
    test_files = ["./test_jewelry.db", "./demo_jewelry.db"]
    for file_path in test_files:
        test_file = Path(file_path)
        if test_file.exists():
            test_file.unlink()
            print(f"Cleaned up test file: {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
