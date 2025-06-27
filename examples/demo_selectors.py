#!/usr/bin/env python3
"""
Demonstration script for eBay Jewelry CSS Selectors Dictionary

This script demonstrates the comprehensive SelectorManager functionality
and provides detailed metrics about the selector system.
"""

import json
from datetime import datetime
from crawl4ai.crawlers.ebay_jewelry.ebay_selectors import (
    SelectorManager, SelectorType, DeviceType, JewelryCategory
)


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("eBay JEWELRY CSS SELECTORS DICTIONARY DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize the SelectorManager
    print("🔧 Initializing SelectorManager...")
    manager = SelectorManager(
        cache_file="./demo_selectors_cache.json",
        enable_analytics=True,
        performance_monitoring=True
    )
    print("✅ SelectorManager initialized successfully!")
    print()
    
    # ============ BASIC METRICS ============
    print("📊 BASIC SYSTEM METRICS")
    print("-" * 40)
    
    selector_counts = manager.get_selector_count()
    total_selectors = sum(selector_counts.values())
    
    print(f"Total Selectors: {total_selectors}")
    print(f"Selector Categories: {len(selector_counts)}")
    print()
    
    print("Breakdown by Type:")
    for selector_type, count in selector_counts.items():
        print(f"  • {selector_type.title().replace('_', ' ')}: {count} selectors")
    print()
    
    # ============ CATEGORIES COVERED ============
    print("💎 JEWELRY CATEGORIES COVERED")
    print("-" * 40)
    
    categories_covered = manager.get_categories_covered()
    print(f"Specialized Categories: {len(categories_covered)}")
    for category in categories_covered:
        print(f"  • {category.title()}")
    print()
    
    # ============ RELIABILITY FEATURES ============
    print("🛡️ RELIABILITY FEATURES")
    print("-" * 40)
    
    reliability_features = manager.get_reliability_features()
    print(f"Primary Selectors: {reliability_features['primary_selectors']}")
    print(f"Fallback Selectors: {reliability_features['fallback_selectors']}")
    print(f"Avg Fallbacks per Selector: {reliability_features['average_fallbacks_per_selector']:.1f}")
    print(f"Success Rate Tracking: {'✅' if reliability_features['success_rate_tracking'] else '❌'}")
    print(f"Performance Monitoring: {'✅' if reliability_features['performance_monitoring'] else '❌'}")
    print(f"Device Responsive: {'✅' if reliability_features['device_responsive'] else '❌'}")
    print(f"Category Specific: {'✅' if reliability_features['category_specific'] else '❌'}")
    print()
    
    # ============ SELECTOR DEMONSTRATIONS ============
    print("🎯 SELECTOR USAGE DEMONSTRATIONS")
    print("-" * 40)
    
    # Example 1: Getting search results selectors
    print("Example 1: Search Results Page Selectors")
    listing_title_selector = manager.get_selector(
        SelectorType.SEARCH_RESULTS, 
        'listing_title'
    )
    if listing_title_selector:
        print(f"  Primary: {listing_title_selector.primary}")
        print(f"  Fallbacks: {len(listing_title_selector.fallbacks)} alternatives")
        print(f"  Description: {listing_title_selector.description}")
    print()
    
    # Example 2: Getting jewelry-specific selectors
    print("Example 2: Jewelry-Specific Selectors")
    metal_type_selector = manager.get_selector(
        SelectorType.JEWELRY_SPECIFIC,
        'metal_type'
    )
    if metal_type_selector:
        print(f"  Primary: {metal_type_selector.primary}")
        print(f"  Fallbacks: {len(metal_type_selector.fallbacks)} alternatives")
        print(f"  Description: {metal_type_selector.description}")
    print()
    
    # Example 3: Mobile-specific selectors
    print("Example 3: Mobile Device Selectors")
    mobile_title_selector = manager.get_selector(
        SelectorType.MOBILE_SPECIFIC,
        'mobile_title',
        device_type=DeviceType.MOBILE
    )
    if mobile_title_selector:
        print(f"  Primary: {mobile_title_selector.primary}")
        print(f"  Fallbacks: {len(mobile_title_selector.fallbacks)} alternatives")
        print(f"  Device Type: {mobile_title_selector.device_type.value}")
    print()
    
    # Example 4: Category-specific selectors
    print("Example 4: Ring-Specific Selectors")
    ring_size_selector = manager.get_selector(
        SelectorType.JEWELRY_SPECIFIC,
        'ring_size',
        category=JewelryCategory.RINGS
    )
    if ring_size_selector:
        print(f"  Primary: {ring_size_selector.primary}")
        print(f"  Category: {ring_size_selector.category_specific.value if ring_size_selector.category_specific else 'General'}")
        print(f"  Description: {ring_size_selector.description}")
    print()
    
    # ============ ALL SELECTORS FOR A FIELD ============
    print("📋 COMPREHENSIVE SELECTOR LISTS")
    print("-" * 40)
    
    print("All possible selectors for 'price' (desktop):")
    price_selectors = manager.get_all_selectors('price', DeviceType.DESKTOP)
    for i, selector in enumerate(price_selectors[:5], 1):  # Show first 5
        print(f"  {i}. {selector}")
    if len(price_selectors) > 5:
        print(f"  ... and {len(price_selectors) - 5} more")
    print()
    
    # ============ PERFORMANCE TRACKING DEMO ============
    print("⚡ PERFORMANCE TRACKING DEMONSTRATION")
    print("-" * 40)
    
    # Simulate some selector usage
    print("Simulating selector usage and performance tracking...")
    
    # Record some sample successes/failures
    manager.record_success(
        SelectorType.SEARCH_RESULTS, 
        'listing_title', 
        'h3.s-item__title', 
        success=True, 
        execution_time=0.05
    )
    
    manager.record_success(
        SelectorType.JEWELRY_SPECIFIC, 
        'metal_type', 
        'td:contains("Metal") + td', 
        success=True, 
        execution_time=0.12
    )
    
    manager.record_success(
        SelectorType.IMAGES, 
        'main_image', 
        '#icImg', 
        success=True, 
        execution_time=0.03
    )
    
    # Generate performance report
    performance_report = manager.get_performance_report()
    
    print(f"Total Selectors Monitored: {performance_report['total_selectors']}")
    print(f"Selector Types: {performance_report['selector_types']}")
    print()
    
    # ============ VALIDATION REPORT ============
    print("✅ VALIDATION REPORT")
    print("-" * 40)
    
    validation_report = manager.validate_selectors()
    print(f"Total Selectors: {validation_report['total_selectors']}")
    print(f"Valid Selectors: {validation_report['valid_selectors']}")
    print(f"Invalid Selectors: {len(validation_report['invalid_selectors'])}")
    print(f"Missing Fallbacks: {len(validation_report['missing_fallbacks'])}")
    print(f"Performance Issues: {len(validation_report['performance_issues'])}")
    
    if validation_report['invalid_selectors']:
        print("\n⚠️  Invalid Selectors:")
        for issue in validation_report['invalid_selectors'][:3]:
            print(f"    • {issue}")
    
    if validation_report['missing_fallbacks']:
        print("\n⚠️  Missing Fallbacks:")
        for issue in validation_report['missing_fallbacks'][:3]:
            print(f"    • {issue}")
    print()
    
    # ============ DETAILED SELECTOR BREAKDOWN ============
    print("📈 DETAILED SELECTOR BREAKDOWN")
    print("-" * 40)
    
    print("Search Results Selectors:")
    search_selectors = manager.selectors[SelectorType.SEARCH_RESULTS]
    for name, selector_info in list(search_selectors.items())[:3]:
        print(f"  • {name}: {selector_info.primary}")
        print(f"    Fallbacks: {len(selector_info.fallbacks)}")
        print(f"    Performance Weight: {selector_info.performance_weight}")
    print()
    
    print("Jewelry-Specific Selectors:")
    jewelry_selectors = manager.selectors[SelectorType.JEWELRY_SPECIFIC]
    for name, selector_info in list(jewelry_selectors.items())[:3]:
        print(f"  • {name}: {selector_info.primary}")
        print(f"    Category: {selector_info.category_specific.value if selector_info.category_specific else 'General'}")
        print(f"    Fallbacks: {len(selector_info.fallbacks)}")
    print()
    
    # ============ EXPORT FUNCTIONALITY ============
    print("💾 CACHE AND EXPORT FUNCTIONALITY")
    print("-" * 40)
    
    print("Saving selector performance cache...")
    manager.save_cache()
    print("✅ Cache saved successfully!")
    
    # Export detailed report
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'total_selectors': total_selectors,
            'categories_covered': categories_covered,
            'reliability_features': reliability_features,
            'selector_counts': selector_counts
        },
        'performance_report': performance_report,
        'validation_report': validation_report
    }
    
    with open('ebay_selectors_report.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("📊 Detailed report exported to 'ebay_selectors_report.json'")
    print()
    
    # ============ SUMMARY ============
    print("📋 FINAL SUMMARY")
    print("=" * 40)
    print(f"✅ Created comprehensive CSS selectors dictionary")
    print(f"📊 Total Selectors: {total_selectors}")
    print(f"🏷️  Categories Covered: {len(categories_covered)} jewelry types")
    print(f"🛡️  Reliability Features: {reliability_features['fallback_selectors']} fallback selectors")
    print(f"📱 Device Support: Desktop, Mobile, Tablet")
    print(f"⚡ Performance Optimization: Built-in")
    print(f"📈 Success Rate Tracking: Enabled")
    print(f"🎯 Jewelry-Specific: Specialized selectors for each category")
    print()
    print("🎉 eBay Jewelry Selector Dictionary is ready for production use!")
    print("=" * 80)


if __name__ == "__main__":
    main()