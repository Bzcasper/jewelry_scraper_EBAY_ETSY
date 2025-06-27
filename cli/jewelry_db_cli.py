#!/usr/bin/env python3
"""
Jewelry Database CLI Tool
Complete command-line interface for jewelry scraping database management.

This tool provides comprehensive database operations including initialization,
data management, querying, export, analytics, validation, and backup operations.
"""

import argparse
import sys
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid

# Import our database manager and models
from ..data.jewelry_data_manager import JewelryDatabaseManager, QueryFilters, DatabaseStats
from ..models.jewelry_models import (
    JewelryListing, JewelryImage, JewelrySpecification, ScrapingSession,
    JewelryCategory, JewelryMaterial, ListingStatus, ImageType
)


class JewelryDatabaseCLI:
    """Command-line interface for jewelry database operations"""

    def __init__(self):
        self.db_manager = None
        self.db_path = "jewelry_scraping.db"

    def setup_database_manager(self, db_path: str = None):
        """Initialize database manager with specified path"""
        if db_path:
            self.db_path = db_path
        self.db_manager = JewelryDatabaseManager(self.db_path)

    def init_database(self, args):
        """Initialize database with schema and indexes"""
        print(f"Initializing database: {self.db_path}")

        success = self.db_manager.initialize_database()

        if success:
            print("✅ Database initialized successfully!")

            # Show initial statistics
            stats = self.db_manager.get_database_stats()
            print(f"Database file: {self.db_path}")
            print(f"Storage size: {stats.storage_size_mb:.2f} MB")

            # Validate database integrity
            print("\n🔍 Validating database integrity...")
            validation = self.db_manager.validate_database_integrity()

            if validation['integrity_check'] and validation['foreign_key_check']:
                print("✅ Database integrity validation passed")
            else:
                print("⚠️  Database integrity issues found:")
                for error in validation['errors']:
                    print(f"   - {error}")
        else:
            print("❌ Failed to initialize database")
            sys.exit(1)

    def show_stats(self, args):
        """Display comprehensive database statistics"""
        print("📊 Database Statistics")
        print("=" * 50)

        stats = self.db_manager.get_database_stats()

        # Basic statistics
        print(f"Total Listings: {stats.total_listings:,}")
        print(f"Total Images: {stats.total_images:,}")
        print(f"Total Specifications: {stats.total_specifications:,}")
        print(f"Total Scraping Sessions: {stats.total_sessions:,}")
        print(f"Average Quality Score: {stats.avg_quality_score:.1%}")
        print(f"Storage Size: {stats.storage_size_mb:.2f} MB")

        if stats.price_range[0] > 0:
            print(
                f"Price Range: ${stats.price_range[0]:.2f} - ${stats.price_range[1]:.2f}")

        # Category breakdown
        if stats.categories_breakdown:
            print("\n📦 Categories:")
            for category, count in sorted(stats.categories_breakdown.items(), key=lambda x: -x[1]):
                percentage = (count / stats.total_listings) * \
                    100 if stats.total_listings > 0 else 0
                print(f"   {category}: {count:,} ({percentage:.1f}%)")

        # Materials breakdown
        if stats.materials_breakdown:
            print("\n💎 Materials:")
            for material, count in sorted(stats.materials_breakdown.items(), key=lambda x: -x[1]):
                percentage = (count / stats.total_listings) * \
                    100 if stats.total_listings > 0 else 0
                print(f"   {material}: {count:,} ({percentage:.1f}%)")

        # Recent activity
        if stats.recent_activity:
            print("\n📈 Recent Activity (Last 7 Days):")
            for date, count in sorted(stats.recent_activity.items()):
                print(f"   {date}: {count:,} listings")

        # Table sizes
        table_sizes = self.db_manager.get_table_sizes()
        print("\n🗂️  Table Sizes:")
        for table, size in table_sizes.items():
            print(f"   {table}: {size:,} rows")

    def query_listings(self, args):
        """Query listings with filters"""
        print("🔍 Querying Listings")
        print("=" * 30)

        # Build filters from arguments
        filters = QueryFilters()

        if args.category:
            filters.category = args.category
        if args.material:
            filters.material = args.material
        if args.min_price:
            filters.min_price = args.min_price
        if args.max_price:
            filters.max_price = args.max_price
        if args.brand:
            filters.brand = args.brand
        if args.seller:
            filters.seller = args.seller
        if args.condition:
            filters.condition = args.condition
        if args.min_quality:
            filters.min_quality_score = args.min_quality
        if args.search:
            filters.search_text = args.search
        if args.validated is not None:
            filters.is_validated = args.validated
        if args.has_images is not None:
            filters.has_images = args.has_images

        # Date filters
        if args.days_back:
            filters.date_from = datetime.now() - timedelta(days=args.days_back)

        # Execute query
        listings = self.db_manager.query_listings(
            filters,
            limit=args.limit or 100,
            offset=args.offset or 0
        )

        if not listings:
            print("No listings found matching criteria")
            return

        print(f"Found {len(listings)} listings")
        print()

        # Display format
        if args.format == 'summary':
            self._display_listings_summary(listings)
        elif args.format == 'detailed':
            self._display_listings_detailed(listings)
        elif args.format == 'json':
            self._display_listings_json(listings)
        else:
            self._display_listings_table(listings)

    def _display_listings_table(self, listings):
        """Display listings in table format"""
        print(f"{'ID':<12} {'Title':<40} {'Price':<12} {'Category':<12} {'Quality':<8}")
        print("-" * 90)

        for listing in listings:
            title = listing.title[:37] + \
                "..." if len(listing.title) > 40 else listing.title
            price = f"${listing.price:.2f}" if listing.price else "N/A"
            quality = f"{listing.data_quality_score:.1%}"

            print(
                f"{listing.id:<12} {title:<40} {price:<12} {listing.category.value:<12} {quality:<8}")

    def _display_listings_summary(self, listings):
        """Display listings summary"""
        for listing in listings:
            summary = listing.get_summary()
            print(f"📦 {summary['title']}")
            print(f"   ID: {summary['id']}")
            print(f"   Price: {summary['price']}")
            print(
                f"   Category: {summary['category']} | Material: {listing.material.value}")
            print(
                f"   Brand: {summary['brand']} | Condition: {summary['condition']}")
            print(
                f"   Seller: {summary['seller']} | Quality: {summary['quality_score']}")
            print(
                f"   Images: {summary['images']} | Scraped: {summary['scraped']}")
            print()

    def _display_listings_detailed(self, listings):
        """Display detailed listing information"""
        for i, listing in enumerate(listings, 1):
            print(f"📦 Listing {i}/{len(listings)}")
            print(f"   ID: {listing.id}")
            print(f"   Title: {listing.title}")
            print(f"   URL: {listing.listing_url}")
            print(f"   Price: ${listing.price:.2f} {listing.currency}")

            if listing.original_price:
                print(f"   Original Price: ${listing.original_price:.2f}")

            print(f"   Category: {listing.category.value}")
            print(f"   Material: {listing.material.value}")

            if listing.brand:
                print(f"   Brand: {listing.brand}")

            print(f"   Condition: {listing.condition}")
            print(f"   Seller: {listing.seller_id}")

            if listing.seller_rating:
                print(f"   Seller Rating: {listing.seller_rating}/100")

            if listing.gemstone:
                print(f"   Gemstone: {listing.gemstone}")

            if listing.size:
                print(f"   Size: {listing.size}")

            print(f"   Images: {listing.image_count}")
            print(f"   Quality Score: {listing.data_quality_score:.1%}")
            print(f"   Validated: {'Yes' if listing.is_validated else 'No'}")
            print(
                f"   Scraped: {listing.scraped_at.strftime('%Y-%m-%d %H:%M:%S')}")

            if listing.description and len(listing.description) > 0:
                desc = listing.description[:200] + "..." if len(
                    listing.description) > 200 else listing.description
                print(f"   Description: {desc}")

            print("-" * 80)

    def _display_listings_json(self, listings):
        """Display listings in JSON format"""
        data = [listing.to_dict(include_metadata=False)
                for listing in listings]
        print(json.dumps(data, indent=2, default=str))

    def export_data(self, args):
        """Export data to various formats"""
        print(f"📤 Exporting data to {args.format.upper()}")

        # Build filters
        filters = QueryFilters()
        if args.category:
            filters.category = args.category
        if args.material:
            filters.material = args.material
        if args.min_price:
            filters.min_price = args.min_price
        if args.max_price:
            filters.max_price = args.max_price
        if args.validated is not None:
            filters.is_validated = args.validated

        # Generate output filename if not provided
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"jewelry_export_{timestamp}.{args.format}"

        # Export based on format
        success = False
        if args.format == 'csv':
            success = self.db_manager.export_to_csv(args.output, filters)
        elif args.format == 'json':
            success = self.db_manager.export_to_json(
                args.output, filters, include_metadata=args.include_metadata)

        if success:
            print(f"✅ Export completed: {args.output}")

            # Show file size
            if os.path.exists(args.output):
                size_mb = os.path.getsize(args.output) / (1024 * 1024)
                print(f"   File size: {size_mb:.2f} MB")
        else:
            print("❌ Export failed")

    def cleanup_data(self, args):
        """Clean up old data"""
        print(f"🧹 Cleaning up data older than {args.days} days")

        # Dry run first to show what would be deleted
        if not args.force:
            print("🔍 Dry run - showing what would be deleted:")
            results = self.db_manager.cleanup_old_data(args.days, dry_run=True)

            print(f"   Listings to delete: {results['listings_to_delete']:,}")
            print(f"   Images to delete: {results['images_to_delete']:,}")

            if results['listings_to_delete'] > 0 or results['images_to_delete'] > 0:
                print(f"\n⚠️  Run with --force to actually delete this data")
            else:
                print("✅ No old data found to clean up")
        else:
            # Actual cleanup
            results = self.db_manager.cleanup_old_data(
                args.days, dry_run=False)

            if results.get('deleted'):
                print(f"✅ Cleanup completed:")
                print(f"   Deleted {results['listings_to_delete']:,} listings")
                print(f"   Deleted {results['images_to_delete']:,} images")

                # Optimize database after cleanup
                print("\n🔧 Optimizing database...")
                self.db_manager.optimize_database()
                print("✅ Database optimization completed")
            else:
                print("ℹ️  No data was cleaned up")

    def validate_database(self, args):
        """Validate database integrity"""
        print("🔍 Validating Database Integrity")
        print("=" * 35)

        validation = self.db_manager.validate_database_integrity()

        # Integrity check
        if validation['integrity_check']:
            print("✅ SQLite integrity check: PASSED")
        else:
            print("❌ SQLite integrity check: FAILED")

        # Foreign key check
        if validation['foreign_key_check']:
            print("✅ Foreign key constraints: PASSED")
        else:
            print("❌ Foreign key constraints: FAILED")

        # Orphaned data
        if validation['orphaned_images'] > 0:
            print(
                f"⚠️  Orphaned images found: {validation['orphaned_images']:,}")
        else:
            print("✅ No orphaned images found")

        if validation['orphaned_specs'] > 0:
            print(
                f"⚠️  Orphaned specifications found: {validation['orphaned_specs']:,}")
        else:
            print("✅ No orphaned specifications found")

        # Duplicates
        if validation['duplicate_listings'] > 0:
            print(
                f"⚠️  Duplicate listings found: {validation['duplicate_listings']:,}")
        else:
            print("✅ No duplicate listings found")

        # Missing required fields
        if validation['missing_required_fields'] > 0:
            print(
                f"⚠️  Listings with missing required fields: {validation['missing_required_fields']:,}")
        else:
            print("✅ All listings have required fields")

        # Errors
        if validation['errors']:
            print(f"\n❌ Validation Errors:")
            for error in validation['errors']:
                print(f"   - {error}")

        if args.fix and (validation['orphaned_images'] > 0 or validation['orphaned_specs'] > 0):
            print("\n🔧 Fixing orphaned data...")
            # Implementation for fixing orphaned data would go here
            print("⚠️  Orphaned data fixing not implemented yet")

    def backup_database(self, args):
        """Create database backup"""
        print("💾 Creating Database Backup")

        try:
            backup_path = self.db_manager.create_backup(args.output)
            print(f"✅ Backup created: {backup_path}")

            # Show backup file size
            size_mb = os.path.getsize(backup_path) / (1024 * 1024)
            print(f"   Backup size: {size_mb:.2f} MB")

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            sys.exit(1)

    def restore_database(self, args):
        """Restore database from backup"""
        print(f"🔄 Restoring Database from {args.backup}")

        if not os.path.exists(args.backup):
            print(f"❌ Backup file not found: {args.backup}")
            sys.exit(1)

        if not args.force:
            print("⚠️  This will overwrite the current database!")
            print("   Add --force to confirm restoration")
            sys.exit(1)

        success = self.db_manager.restore_backup(args.backup)

        if success:
            print("✅ Database restored successfully")

            # Show statistics of restored database
            print("\n📊 Restored Database Statistics:")
            stats = self.db_manager.get_database_stats()
            print(f"   Total listings: {stats.total_listings:,}")
            print(f"   Total images: {stats.total_images:,}")
            print(f"   Storage size: {stats.storage_size_mb:.2f} MB")
        else:
            print("❌ Database restoration failed")
            sys.exit(1)

    def optimize_database(self, args):
        """Optimize database performance"""
        print("🔧 Optimizing Database Performance")

        success = self.db_manager.optimize_database()

        if success:
            print("✅ Database optimization completed")

            # Show updated statistics
            stats = self.db_manager.get_database_stats()
            print(f"   Storage size: {stats.storage_size_mb:.2f} MB")
        else:
            print("❌ Database optimization failed")

    def create_sample_data(self, args):
        """Create sample data for testing"""
        print(f"🧪 Creating {args.count} sample listings")

        sample_listings = []

        for i in range(args.count):
            listing = JewelryListing(
                id=f"sample_{uuid.uuid4().hex[:8]}",
                title=f"Sample Jewelry Item #{i+1}",
                price=round(100 + (i * 50) + (i % 1000), 2),
                currency="USD",
                condition="New",
                seller_id=f"seller_{(i % 10) + 1}",
                listing_url=f"https://ebay.com/itm/sample_{i+1}",
                end_time=None,
                shipping_cost=None,
                category=list(JewelryCategory)[i % len(JewelryCategory)],
                material=list(JewelryMaterial)[i % len(JewelryMaterial)],
                gemstone="Diamond" if i % 4 == 0 else None,
                size=None,
                weight=None,
                brand=f"Brand {(i % 5) + 1}" if i % 3 == 0 else None,
                main_image_path=None,
                listing_id=None,
                original_price=None,
                availability=None,
                seller_rating=None,
                seller_feedback_count=None,
                subcategory=None,
                dimensions=None,
                stone_color=None,
                stone_clarity=None,
                stone_cut=None,
                stone_carat=None,
                description=f"This is a sample jewelry item #{i+1} for testing purposes.",
                item_number=None,
                listing_type=None,
                watchers=None,
                views=None,
                bids=None,
                time_left=None,
                ships_from=None,
                ships_to=None,
                listing_date=None,
                image_count=i % 8,
                data_quality_score=0.5 + (0.5 * (i % 10) / 10)
            )

            listing.update_quality_score()
            sample_listings.append(listing)

        # Batch insert
        inserted = self.db_manager.batch_insert_listings(sample_listings)

        print(f"✅ Created {inserted} sample listings")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Jewelry Database Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jewelry_db_cli.py init                              # Initialize database
  jewelry_db_cli.py stats                             # Show statistics
  jewelry_db_cli.py query --category rings --min-price 100
  jewelry_db_cli.py export --format csv --output rings.csv
  jewelry_db_cli.py cleanup --days 30                 # Dry run cleanup
  jewelry_db_cli.py cleanup --days 30 --force         # Actually cleanup
  jewelry_db_cli.py validate                          # Check integrity
  jewelry_db_cli.py backup --output backup.db         # Create backup
  jewelry_db_cli.py optimize                          # Optimize performance
        """)

    parser.add_argument('--database', '-d', default='jewelry_scraping.db',
                        help='Database file path (default: jewelry_scraping.db)')

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize database')

    # Statistics command
    stats_parser = subparsers.add_parser(
        'stats', help='Show database statistics')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query listings')
    query_parser.add_argument(
        '--category', choices=[c.value for c in JewelryCategory])
    query_parser.add_argument(
        '--material', choices=[m.value for m in JewelryMaterial])
    query_parser.add_argument('--min-price', type=float)
    query_parser.add_argument('--max-price', type=float)
    query_parser.add_argument('--brand')
    query_parser.add_argument('--seller')
    query_parser.add_argument('--condition')
    query_parser.add_argument(
        '--min-quality', type=float, help='Minimum quality score (0-1)')
    query_parser.add_argument(
        '--search', help='Search in title and description')
    query_parser.add_argument('--validated', type=bool,
                              help='Filter by validation status')
    query_parser.add_argument(
        '--has-images', type=bool, help='Filter by image presence')
    query_parser.add_argument('--days-back', type=int,
                              help='Only show listings from last N days')
    query_parser.add_argument('--limit', type=int, default=100)
    query_parser.add_argument('--offset', type=int, default=0)
    query_parser.add_argument('--format', choices=['table', 'summary', 'detailed', 'json'],
                              default='table', help='Output format')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument(
        '--format', choices=['csv', 'json'], required=True)
    export_parser.add_argument('--output', help='Output file path')
    export_parser.add_argument(
        '--category', choices=[c.value for c in JewelryCategory])
    export_parser.add_argument(
        '--material', choices=[m.value for m in JewelryMaterial])
    export_parser.add_argument('--min-price', type=float)
    export_parser.add_argument('--max-price', type=float)
    export_parser.add_argument(
        '--validated', type=bool, help='Filter by validation status')
    export_parser.add_argument('--include-metadata', action='store_true',
                               help='Include metadata in JSON export')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument(
        '--days', type=int, default=30, help='Delete data older than N days')
    cleanup_parser.add_argument(
        '--force', action='store_true', help='Actually perform cleanup')

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate', help='Validate database integrity')
    validate_parser.add_argument(
        '--fix', action='store_true', help='Fix issues if possible')

    # Backup command
    backup_parser = subparsers.add_parser(
        'backup', help='Create database backup')
    backup_parser.add_argument('--output', help='Backup file path')

    # Restore command
    restore_parser = subparsers.add_parser(
        'restore', help='Restore from backup')
    restore_parser.add_argument('backup', help='Backup file path')
    restore_parser.add_argument(
        '--force', action='store_true', help='Confirm restoration')

    # Optimize command
    optimize_parser = subparsers.add_parser(
        'optimize', help='Optimize database performance')

    # Sample data command (for testing)
    sample_parser = subparsers.add_parser(
        'sample', help='Create sample data for testing')
    sample_parser.add_argument(
        '--count', type=int, default=100, help='Number of sample listings')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = JewelryDatabaseCLI()
    cli.setup_database_manager(args.database)

    # Execute command
    try:
        if args.command == 'init':
            cli.init_database(args)
        elif args.command == 'stats':
            cli.show_stats(args)
        elif args.command == 'query':
            cli.query_listings(args)
        elif args.command == 'export':
            cli.export_data(args)
        elif args.command == 'cleanup':
            cli.cleanup_data(args)
        elif args.command == 'validate':
            cli.validate_database(args)
        elif args.command == 'backup':
            cli.backup_database(args)
        elif args.command == 'restore':
            cli.restore_database(args)
        elif args.command == 'optimize':
            cli.optimize_database(args)
        elif args.command == 'sample':
            cli.create_sample_data(args)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
