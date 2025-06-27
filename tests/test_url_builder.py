#!/usr/bin/env python3
"""
Test and demonstration script for the eBay URL Builder

This script demonstrates the capabilities of the eBay URL Builder
and provides examples of different search configurations.
"""

from ..scrapers.ebay_url_builder import (
    EBayURLBuilder, SearchFilters, SortOrder, ItemCondition,
    ListingType, JewelryCategory, MetalType, GemstoneMaterial,
    build_jewelry_search_url, build_mobile_jewelry_url
)


def demo_basic_search():
    """Demonstrate basic search URL building"""
    print("=== Basic Search Demo ===")

    # Simple search
    url = build_jewelry_search_url("diamond ring", JewelryCategory.RINGS)
    print(f"Simple search: {url}")

    # Search with price range
    url = build_jewelry_search_url(
        "gold necklace",
        JewelryCategory.NECKLACES,
        min_price=100.0,
        max_price=500.0,
        condition=[ItemCondition.NEW, ItemCondition.NEW_OTHER]
    )
    print(f"With price range: {url}")
    print()


def demo_advanced_search():
    """Demonstrate advanced search with multiple filters"""
    print("=== Advanced Search Demo ===")

    builder = EBayURLBuilder(enable_validation=True)

    # Complex jewelry search
    filters = SearchFilters(
        query="vintage engagement ring",
        category=JewelryCategory.RINGS,
        min_price=500.0,
        max_price=2000.0,
        condition=[ItemCondition.NEW, ItemCondition.USED],
        listing_type=ListingType.BUY_IT_NOW,
        sort_order=SortOrder.PRICE_LOW_TO_HIGH,
        metal_type=MetalType.GOLD,
        gemstone_material=GemstoneMaterial.DIAMOND,
        ring_size="7",
        free_shipping_only=True,
        top_rated_sellers_only=True,
        items_per_page=100
    )

    result = builder.build_search_url(filters)

    print(f"Advanced search URL: {result.url}")
    print(f"Valid: {result.is_valid}")
    print(f"Parameters: {result.parameter_count}")
    print(f"Estimated results: {result.estimated_results}")

    if result.warnings:
        print(f"Warnings: {result.warnings}")
    if result.errors:
        print(f"Errors: {result.errors}")
    print()


def demo_mobile_optimization():
    """Demonstrate mobile URL building"""
    print("=== Mobile Optimization Demo ===")

    # Mobile-optimized URL
    mobile_url = build_mobile_jewelry_url(
        "pearl earrings", JewelryCategory.EARRINGS)
    print(f"Mobile URL: {mobile_url}")

    # Compare desktop vs mobile
    desktop_builder = EBayURLBuilder(mobile_mode=False)
    mobile_builder = EBayURLBuilder(mobile_mode=True)

    filters = SearchFilters(
        query="silver bracelet",
        category=JewelryCategory.BRACELETS,
        max_price=100.0
    )

    desktop_result = desktop_builder.build_search_url(filters)
    mobile_result = mobile_builder.build_search_url(filters)

    print(f"Desktop: {desktop_result.url}")
    print(f"Mobile:  {mobile_result.url}")
    print()


def demo_category_browsing():
    """Demonstrate category-specific URL building"""
    print("=== Category Browsing Demo ===")

    builder = EBayURLBuilder()

    # Browse different jewelry categories
    categories = [
        JewelryCategory.RINGS,
        JewelryCategory.NECKLACES,
        JewelryCategory.WATCHES,
        JewelryCategory.GEMSTONES
    ]

    for category in categories:
        url = builder.build_category_url(category)
        print(f"{category.value['name']}: {url}")

    # Browse with subcategory
    url = builder.build_category_url(JewelryCategory.RINGS, 'engagement')
    print(f"Engagement rings: {url}")
    print()


def demo_seller_search():
    """Demonstrate seller-specific searches"""
    print("=== Seller Search Demo ===")

    builder = EBayURLBuilder()

    # Search specific seller
    url = builder.build_seller_url("tiffanyandco", JewelryCategory.RINGS)
    print(f"Tiffany & Co rings: {url}")

    # General seller search
    url = builder.build_seller_url("bluenile")
    print(f"Blue Nile all items: {url}")
    print()


def demo_validation_features():
    """Demonstrate URL validation capabilities"""
    print("=== Validation Features Demo ===")

    builder = EBayURLBuilder(enable_validation=True, max_url_length=1000)

    # Create a potentially problematic search
    filters = SearchFilters(
        query="very long search query with many words that might make URL too long",
        category=JewelryCategory.RINGS,
        min_price=1000.0,
        max_price=500.0,  # Invalid: min > max
        condition=[ItemCondition.NEW,
                   ItemCondition.USED, ItemCondition.FOR_PARTS],
        listing_type=ListingType.AUCTION,
        buy_it_now_only=True,  # Conflicting with auction
        metal_type=MetalType.GOLD,
        gemstone_material=GemstoneMaterial.DIAMOND,
        brand="Some Very Long Brand Name That Might Cause Issues",
        page_number=150,  # Very high page number
        items_per_page=250,  # Exceeds eBay limit
        extra_params={
            'custom_param_1': 'value1',
            'custom_param_2': 'value2',
            'custom_param_3': 'value3'
        }
    )

    result = builder.build_search_url(filters)

    print(f"Validation result: {result.is_valid}")
    print(f"URL length: {len(result.url)} characters")
    print(f"Parameters: {result.parameter_count}")

    if result.errors:
        print("Errors found:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    print()


def demo_parameter_overview():
    """Show overview of supported parameters"""
    print("=== Parameter Overview ===")

    builder = EBayURLBuilder()

    # Get supported parameters
    params = builder.get_supported_parameters()

    for category, parameters in params.items():
        print(f"{category.upper()}:")
        for param, description in parameters.items():
            print(f"  {param}: {description}")
        print()

    # Get category mappings
    print("CATEGORY MAPPINGS:")
    mappings = builder.get_category_mappings()
    for category_name, info in mappings.items():
        print(
            f"  {category_name}: ID {info['category_id']} - {info['category_name']}")
        if info['subcategories']:
            for subcat, subcat_id in info['subcategories'].items():
                print(f"    └─ {subcat}: {subcat_id}")
    print()


def demo_statistics():
    """Show builder usage statistics"""
    print("=== Usage Statistics ===")

    builder = EBayURLBuilder()

    # Build several URLs to generate stats
    test_searches = [
        SearchFilters(query="diamond", category=JewelryCategory.RINGS),
        SearchFilters(
            query="gold", category=JewelryCategory.NECKLACES, min_price=100.0),
        SearchFilters(query="silver", metal_type=MetalType.SILVER),
        SearchFilters(category=JewelryCategory.WATCHES, max_price=500.0)
    ]

    for filters in test_searches:
        builder.build_search_url(filters)

    stats = builder.get_statistics()

    print(f"URLs built: {stats['urls_built']}")
    print(f"Validation errors: {stats['validation_errors']}")

    print("Most used parameters:")
    sorted_params = sorted(
        stats['parameters_used'].items(), key=lambda x: x[1], reverse=True)
    for param, count in sorted_params[:10]:
        print(f"  {param}: {count} times")
    print()


def main():
    """Run all demonstrations"""
    print("eBay URL Builder Demonstration")
    print("=" * 50)
    print()

    demo_basic_search()
    demo_advanced_search()
    demo_mobile_optimization()
    demo_category_browsing()
    demo_seller_search()
    demo_validation_features()
    demo_parameter_overview()
    demo_statistics()

    print("Demonstration complete!")


if __name__ == "__main__":
    main()
