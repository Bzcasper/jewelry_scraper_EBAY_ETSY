#!/usr/bin/env python3
"""
Integration Test for Jewelry Scraping System
Tests end-to-end functionality of all major components
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path


async def test_jewelry_models():
    """Test jewelry models functionality"""
    print("Testing Jewelry Models...")

    try:
        from ..models.jewelry_models import (
            JewelryListing, JewelryCategory, JewelryMaterial,
            ListingStatus, JewelryImage, ImageType
        )

        # Create a test listing
        listing = JewelryListing(
            id="test_123",
            title="Beautiful Diamond Ring",
            price=500.0,
            currency="USD",
            condition="New",
            seller_id="test_seller",
            listing_url="https://www.ebay.com/itm/Beautiful-Diamond-Ring-14K-Gold-Solitaire/155777888999",
            category=JewelryCategory.RINGS,
            material=JewelryMaterial.GOLD
        )

        # Test validation
        assert listing.validate_for_database() == True

        # Test quality score calculation
        listing.update_quality_score()
        assert listing.data_quality_score > 0

        # Test JSON serialization
        json_data = listing.to_json()
        assert "test_123" in json_data

        print("✓ Jewelry Models test passed")
        return True

    except Exception as e:
        print(f"✗ Jewelry Models test failed: {e}")
        return False


async def test_extraction_pipeline():
    """Test the main extraction pipeline components"""
    print("Testing Extraction Pipeline...")

    try:
        from ..core.jewelry_extraction_pipeline import JewelryExtractor

        # Create extractor with default settings (no config needed)
        extractor = JewelryExtractor()

        # Test initialization
        assert extractor is not None

        print("✓ Extraction Pipeline test passed")
        return True

    except Exception as e:
        print(f"✗ Extraction Pipeline test failed: {e}")
        return False


async def test_mcp_server():
    """Test MCP server components (optional - requires deploy/docker setup)"""
    print("Testing MCP Server...")

    try:
        # Try to import from deploy/docker if available
        try:
            from deploy.docker.mcp_bridge import attach_mcp, mcp_tool
            from fastapi import FastAPI
        except ImportError:
            print("⚠️  MCP server components not available (deploy/docker not found)")
            return

        # Create test FastAPI app
        app = FastAPI(title="Test App")

        @app.get("/test")
        @mcp_tool("test_tool")
        async def test_endpoint():
            return {"status": "ok"}

        # Test MCP attachment
        attach_mcp(app, base_url="http://localhost:8000")

        print("✓ MCP Server test passed")
        return True

    except Exception as e:
        print(f"✗ MCP Server test failed: {e}")
        return False


async def test_database_operations():
    """Test database operations with SQLite"""
    print("Testing Database Operations...")

    try:
        import aiosqlite
        import tempfile
        from ..models.jewelry_models import JEWELRY_SCHEMA_SQL, JEWELRY_INDEXES_SQL

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            # Test database creation
            async with aiosqlite.connect(db_path) as db:
                # Create tables
                for table_name, schema in JEWELRY_SCHEMA_SQL.items():
                    await db.execute(schema)

                # Create indexes
                for index_sql in JEWELRY_INDEXES_SQL:
                    await db.execute(index_sql)

                await db.commit()

                # Test basic insert
                await db.execute("""
                    INSERT INTO jewelry_listings 
                    (listing_id, url, title, price, currency, condition, category, material) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("test_123", "https://test.com", "Test Ring", 100.0, "USD", "New", "rings", "gold"))

                await db.commit()

                # Test basic query
                async with db.execute("SELECT COUNT(*) FROM jewelry_listings") as cursor:
                    count = await cursor.fetchone()
                    assert count[0] == 1

            print("✓ Database Operations test passed")
            return True

        finally:
            # Clean up temporary database
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print(f"✗ Database Operations test failed: {e}")
        return False


async def test_image_processing():
    """Test image processing capabilities"""
    print("Testing Image Processing...")

    try:
        from PIL import Image
        import imagehash
        import tempfile

        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_path = tmp.name

        try:
            # Create a simple RGB image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(img_path)

            # Test image hash
            with Image.open(img_path) as test_img:
                hash_val = imagehash.average_hash(test_img)
                assert hash_val is not None

            print("✓ Image Processing test passed")
            return True

        finally:
            if os.path.exists(img_path):
                os.unlink(img_path)

    except Exception as e:
        print(f"✗ Image Processing test failed: {e}")
        return False


async def test_authentication():
    """Test JWT authentication system (optional - requires deploy/docker setup)"""
    print("Testing Authentication...")

    try:
        # Try to import from deploy/docker if available
        try:
            from deploy.docker.auth import create_access_token, verify_token
            from fastapi.security import HTTPAuthorizationCredentials
        except ImportError:
            print("⚠️  Authentication components not available (deploy/docker not found)")
            return

        # Test token creation
        test_data = {"user_id": "test_user", "email": "test@example.com"}
        token = create_access_token(test_data)
        assert token is not None
        assert len(token) > 10

        print("✓ Authentication test passed")
        return True

    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("Starting Jewelry Scraping System Integration Tests")
    print("=" * 60)

    tests = [
        test_jewelry_models,
        test_extraction_pipeline,
        test_mcp_server,
        test_database_operations,
        test_image_processing,
        test_authentication,
    ]

    results = []

    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 60)
    print(f"Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed! System is fully functional.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
