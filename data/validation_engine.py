"""
Advanced Validation Engine for Jewelry Database
Provides comprehensive data validation, integrity checking, and quality assurance.
"""

import re
import sqlite3
import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import hashlib

from .database_manager import DatabaseManager
from .models import JewelryListing, JewelryImage, JewelrySpecification, JewelryCategory, JewelryMaterial

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    BASIC = "basic"           # Essential field validation
    STANDARD = "standard"     # Standard business rules
    STRICT = "strict"         # Comprehensive validation
    PEDANTIC = "pedantic"     # Maximum validation


class ValidationCategory(str, Enum):
    """Categories of validation rules"""
    REQUIRED_FIELDS = "required_fields"
    DATA_TYPES = "data_types"
    FORMAT_VALIDATION = "format_validation"
    BUSINESS_RULES = "business_rules"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    DATA_QUALITY = "data_quality"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Represents a single validation rule"""

    rule_id: str
    name: str
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    check_function: callable
    required_level: ValidationLevel = ValidationLevel.STANDARD
    table: Optional[str] = None
    field_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """Represents a validation issue found"""

    rule_id: str
    severity: ValidationSeverity
    category: ValidationCategory
    table: Optional[str]
    field: Optional[str]
    record_id: Optional[str]
    message: str
    current_value: Any = None
    suggested_value: Any = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    found_at: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""

    validation_id: str
    level: ValidationLevel
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Summary statistics
    total_records_checked: int = 0
    total_issues_found: int = 0
    issues_by_severity: Dict[ValidationSeverity,
                             int] = field(default_factory=dict)
    issues_by_category: Dict[ValidationCategory,
                             int] = field(default_factory=dict)

    # Detailed issues
    issues: List[ValidationIssue] = field(default_factory=list)

    # Performance metrics
    execution_time_seconds: Optional[float] = None
    rules_executed: int = 0

    # Quality scores
    overall_quality_score: Optional[float] = None
    table_quality_scores: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report"""
        self.issues.append(issue)
        self.total_issues_found += 1

        # Update counters
        self.issues_by_severity[issue.severity] = self.issues_by_severity.get(
            issue.severity, 0) + 1
        self.issues_by_category[issue.category] = self.issues_by_category.get(
            issue.category, 0) + 1


class ValidationEngine:
    """
    Advanced validation engine for comprehensive data quality assurance
    """

    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize validation engine

        Args:
            database_manager: DatabaseManager instance
        """
        self.db_manager = database_manager
        self.validation_rules: List[ValidationRule] = []
        self._init_validation_rules()

        # Validation cache for performance
        self._validation_cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 hour

        logger.info("ValidationEngine initialized with {} rules".format(
            len(self.validation_rules)))

    def _init_validation_rules(self):
        """Initialize all validation rules"""

        # Required fields validation
        self._add_required_field_rules()

        # Data type validation
        self._add_data_type_rules()

        # Format validation
        self._add_format_validation_rules()

        # Business rules validation
        self._add_business_rules()

        # Referential integrity rules
        self._add_referential_integrity_rules()

        # Data quality rules
        self._add_data_quality_rules()

        # Consistency rules
        self._add_consistency_rules()

        # Completeness rules
        self._add_completeness_rules()

    def _add_required_field_rules(self):
        """Add rules for required field validation"""

        # Jewelry listings required fields
        required_fields = {
            'jewelry_listings': [
                ('listing_id', 'Listing ID is required'),
                ('title', 'Title is required'),
                ('price', 'Price is required'),
                ('currency', 'Currency is required'),
                ('condition', 'Condition is required'),
                ('seller_name', 'Seller name is required'),
                ('category', 'Category is required'),
                ('material', 'Material is required'),
                ('url', 'Listing URL is required')
            ]
        }

        for table, fields in required_fields.items():
            for field, message in fields:
                self.validation_rules.append(ValidationRule(
                    rule_id=f"required_{table}_{field}",
                    name=f"Required Field: {field}",
                    category=ValidationCategory.REQUIRED_FIELDS,
                    severity=ValidationSeverity.ERROR,
                    description=message,
                    check_function=lambda table=table, field_name=field, msg=message: self._check_required_field(
                        table, field, msg),
                    required_level=ValidationLevel.BASIC,
                    table=table,
                    field_name=field
                ))

    def _add_data_type_rules(self):
        """Add data type validation rules"""

        # Price validation
        self.validation_rules.append(ValidationRule(
            rule_id="datatype_price_positive",
            name="Price Must Be Positive",
            category=ValidationCategory.DATA_TYPES,
            severity=ValidationSeverity.ERROR,
            description="Price must be a positive number",
            check_function=self._check_positive_price,
            required_level=ValidationLevel.BASIC,
            table="jewelry_listings",
            field_name="price"
        ))

        # Quality score validation
        self.validation_rules.append(ValidationRule(
            rule_id="datatype_quality_score_range",
            name="Quality Score Range",
            category=ValidationCategory.DATA_TYPES,
            severity=ValidationSeverity.WARNING,
            description="Quality score must be between 0 and 1",
            check_function=self._check_quality_score_range,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings",
            field_name="data_completeness_score"
        ))

        # Image count validation
        self.validation_rules.append(ValidationRule(
            rule_id="datatype_image_count_nonnegative",
            name="Image Count Non-negative",
            category=ValidationCategory.DATA_TYPES,
            severity=ValidationSeverity.WARNING,
            description="Image count must be non-negative",
            check_function=self._check_nonnegative_image_count,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings",
            field_name="image_count"
        ))

    def _add_format_validation_rules(self):
        """Add format validation rules"""

        # URL format validation
        self.validation_rules.append(ValidationRule(
            rule_id="format_url_valid",
            name="Valid URL Format",
            category=ValidationCategory.FORMAT_VALIDATION,
            severity=ValidationSeverity.ERROR,
            description="URL must be in valid format",
            check_function=self._check_url_format,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings",
            field_name="url"
        ))

        # Currency code validation
        self.validation_rules.append(ValidationRule(
            rule_id="format_currency_iso",
            name="ISO Currency Code",
            category=ValidationCategory.FORMAT_VALIDATION,
            severity=ValidationSeverity.WARNING,
            description="Currency should be valid ISO code",
            check_function=self._check_currency_format,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings",
            field_name="currency"
        ))

        # Email format validation (if present)
        self.validation_rules.append(ValidationRule(
            rule_id="format_email_valid",
            name="Valid Email Format",
            category=ValidationCategory.FORMAT_VALIDATION,
            severity=ValidationSeverity.WARNING,
            description="Email must be in valid format",
            check_function=self._check_email_format,
            required_level=ValidationLevel.STRICT,
            table="jewelry_listings",
            field_name="seller_email"
        ))

    def _add_business_rules(self):
        """Add business logic validation rules"""

        # Price reasonableness
        self.validation_rules.append(ValidationRule(
            rule_id="business_price_reasonable",
            name="Reasonable Price Range",
            category=ValidationCategory.BUSINESS_RULES,
            severity=ValidationSeverity.WARNING,
            description="Price should be within reasonable range for jewelry",
            check_function=self._check_reasonable_price,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings",
            field_name="price"
        ))

        # Original price logic
        self.validation_rules.append(ValidationRule(
            rule_id="business_original_price_logic",
            name="Original Price Logic",
            category=ValidationCategory.BUSINESS_RULES,
            severity=ValidationSeverity.WARNING,
            description="Original price should be greater than or equal to current price",
            check_function=self._check_original_price_logic,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings"
        ))

        # Image count vs listing quality
        self.validation_rules.append(ValidationRule(
            rule_id="business_images_quality_correlation",
            name="Images-Quality Correlation",
            category=ValidationCategory.BUSINESS_RULES,
            severity=ValidationSeverity.INFO,
            description="Listings with more images typically have higher quality scores",
            check_function=self._check_images_quality_correlation,
            required_level=ValidationLevel.STRICT,
            table="jewelry_listings"
        ))

    def _add_referential_integrity_rules(self):
        """Add referential integrity validation rules"""

        # Orphaned images
        self.validation_rules.append(ValidationRule(
            rule_id="integrity_orphaned_images",
            name="Orphaned Images",
            category=ValidationCategory.REFERENTIAL_INTEGRITY,
            severity=ValidationSeverity.ERROR,
            description="Images should reference valid listings",
            check_function=self._check_orphaned_images,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_images"
        ))

        # Orphaned specifications
        self.validation_rules.append(ValidationRule(
            rule_id="integrity_orphaned_specs",
            name="Orphaned Specifications",
            category=ValidationCategory.REFERENTIAL_INTEGRITY,
            severity=ValidationSeverity.ERROR,
            description="Specifications should reference valid listings",
            check_function=self._check_orphaned_specifications,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_specifications"
        ))

    def _add_data_quality_rules(self):
        """Add data quality validation rules"""

        # Title quality
        self.validation_rules.append(ValidationRule(
            rule_id="quality_title_descriptive",
            name="Descriptive Title",
            category=ValidationCategory.DATA_QUALITY,
            severity=ValidationSeverity.WARNING,
            description="Title should be descriptive and informative",
            check_function=self._check_title_quality,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings",
            field_name="title"
        ))

        # Description completeness
        self.validation_rules.append(ValidationRule(
            rule_id="quality_description_completeness",
            name="Description Completeness",
            category=ValidationCategory.DATA_QUALITY,
            severity=ValidationSeverity.INFO,
            description="Descriptions should be sufficiently detailed",
            check_function=self._check_description_completeness,
            required_level=ValidationLevel.STRICT,
            table="jewelry_listings",
            field_name="description"
        ))

        # Duplicate detection
        self.validation_rules.append(ValidationRule(
            rule_id="quality_duplicate_listings",
            name="Duplicate Listings",
            category=ValidationCategory.DATA_QUALITY,
            severity=ValidationSeverity.WARNING,
            description="Check for potential duplicate listings",
            check_function=self._check_duplicate_listings,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings"
        ))

    def _add_consistency_rules(self):
        """Add data consistency validation rules"""

        # Category-material consistency
        self.validation_rules.append(ValidationRule(
            rule_id="consistency_category_material",
            name="Category-Material Consistency",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.WARNING,
            description="Material should be appropriate for category",
            check_function=self._check_category_material_consistency,
            required_level=ValidationLevel.STRICT,
            table="jewelry_listings"
        ))

        # Price-category consistency
        self.validation_rules.append(ValidationRule(
            rule_id="consistency_price_category",
            name="Price-Category Consistency",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.INFO,
            description="Price should be typical for category",
            check_function=self._check_price_category_consistency,
            required_level=ValidationLevel.STRICT,
            table="jewelry_listings"
        ))

    def _add_completeness_rules(self):
        """Add data completeness validation rules"""

        # Essential fields completeness
        self.validation_rules.append(ValidationRule(
            rule_id="completeness_essential_fields",
            name="Essential Fields Completeness",
            category=ValidationCategory.COMPLETENESS,
            severity=ValidationSeverity.WARNING,
            description="Essential fields should be populated",
            check_function=self._check_essential_fields_completeness,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings"
        ))

        # Image availability
        self.validation_rules.append(ValidationRule(
            rule_id="completeness_images_available",
            name="Images Available",
            category=ValidationCategory.COMPLETENESS,
            severity=ValidationSeverity.INFO,
            description="Listings should have at least one image",
            check_function=self._check_images_available,
            required_level=ValidationLevel.STANDARD,
            table="jewelry_listings"
        ))

    # === VALIDATION CHECK FUNCTIONS ===

    def _check_required_field(self, table: str, field: str, message: str) -> List[ValidationIssue]:
        """Check for required field violations"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT listing_id, {field}
                FROM {table}
                WHERE {field} IS NULL OR {field} = ''
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id=f"required_{table}_{field}",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.REQUIRED_FIELDS,
                    table=table,
                    field_name=field,
                    record_id=row[0],
                    message=message,
                    current_value=row[1]
                ))

        return issues

    def _check_positive_price(self) -> List[ValidationIssue]:
        """Check for non-positive prices"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, price
                FROM jewelry_listings
                WHERE price IS NOT NULL AND price <= 0
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="datatype_price_positive",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATA_TYPES,
                    table="jewelry_listings",
                    field_name="price",
                    record_id=row[0],
                    message=f"Price must be positive, found: {row[1]}",
                    current_value=row[1],
                    suggested_value=None
                ))

        return issues

    def _check_quality_score_range(self) -> List[ValidationIssue]:
        """Check quality score is within valid range"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, data_completeness_score
                FROM jewelry_listings
                WHERE data_completeness_score IS NOT NULL 
                AND (data_completeness_score < 0 OR data_completeness_score > 1)
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="datatype_quality_score_range",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DATA_TYPES,
                    table="jewelry_listings",
                    field_name="data_completeness_score",
                    record_id=row[0],
                    message=f"Quality score must be between 0 and 1, found: {row[1]}",
                    current_value=row[1],
                    suggested_value=max(
                        0, min(1, row[1])) if row[1] is not None else None
                ))

        return issues

    def _check_nonnegative_image_count(self) -> List[ValidationIssue]:
        """Check image count is non-negative"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, image_count
                FROM jewelry_listings
                WHERE image_count IS NOT NULL AND image_count < 0
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="datatype_image_count_nonnegative",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DATA_TYPES,
                    table="jewelry_listings",
                    field_name="image_count",
                    record_id=row[0],
                    message=f"Image count must be non-negative, found: {row[1]}",
                    current_value=row[1],
                    suggested_value=0
                ))

        return issues

    def _check_url_format(self) -> List[ValidationIssue]:
        """Check URL format validity"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, url
                FROM jewelry_listings
                WHERE url IS NOT NULL AND url != ''
            """)

            for row in cursor.fetchall():
                try:
                    parsed = urlparse(row[1])
                    if not parsed.scheme or not parsed.netloc:
                        issues.append(ValidationIssue(
                            rule_id="format_url_valid",
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.FORMAT_VALIDATION,
                            table="jewelry_listings",
                            field_name="url",
                            record_id=row[0],
                            message=f"Invalid URL format: {row[1]}",
                            current_value=row[1]
                        ))
                except Exception:
                    issues.append(ValidationIssue(
                        rule_id="format_url_valid",
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.FORMAT_VALIDATION,
                        table="jewelry_listings",
                        field_name="url",
                        record_id=row[0],
                        message=f"Malformed URL: {row[1]}",
                        current_value=row[1]
                    ))

        return issues

    def _check_currency_format(self) -> List[ValidationIssue]:
        """Check currency code format"""
        issues = []
        valid_currencies = {'USD', 'EUR', 'GBP',
                            'CAD', 'AUD', 'JPY', 'CHF', 'CNY'}

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, currency
                FROM jewelry_listings
                WHERE currency IS NOT NULL AND currency != ''
            """)

            for row in cursor.fetchall():
                if len(row[1]) != 3 or not row[1].isupper():
                    issues.append(ValidationIssue(
                        rule_id="format_currency_iso",
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.FORMAT_VALIDATION,
                        table="jewelry_listings",
                        field_name="currency",
                        record_id=row[0],
                        message=f"Currency should be 3-letter uppercase ISO code, found: {row[1]}",
                        current_value=row[1],
                        suggested_value=row[1].upper() if isinstance(
                            row[1], str) else None
                    ))
                elif row[1] not in valid_currencies:
                    issues.append(ValidationIssue(
                        rule_id="format_currency_iso",
                        severity=ValidationSeverity.INFO,
                        category=ValidationCategory.FORMAT_VALIDATION,
                        table="jewelry_listings",
                        field_name="currency",
                        record_id=row[0],
                        message=f"Uncommon currency code: {row[1]}",
                        current_value=row[1]
                    ))

        return issues

    def _check_email_format(self) -> List[ValidationIssue]:
        """Check email format validity"""
        issues = []
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Check if email field exists
            try:
                cursor.execute("""
                    SELECT listing_id, seller_email
                    FROM jewelry_listings
                    WHERE seller_email IS NOT NULL AND seller_email != ''
                """)

                for row in cursor.fetchall():
                    if not email_pattern.match(row[1]):
                        issues.append(ValidationIssue(
                            rule_id="format_email_valid",
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.FORMAT_VALIDATION,
                            table="jewelry_listings",
                            field_name="seller_email",
                            record_id=row[0],
                            message=f"Invalid email format: {row[1]}",
                            current_value=row[1]
                        ))
            except sqlite3.OperationalError:
                # Email field doesn't exist, skip this validation
                pass

        return issues

    def _check_reasonable_price(self) -> List[ValidationIssue]:
        """Check if prices are within reasonable ranges"""
        issues = []

        # Define reasonable price ranges by category
        price_ranges = {
            'rings': (1, 50000),
            'necklaces': (5, 30000),
            'earrings': (5, 20000),
            'bracelets': (10, 25000),
            'watches': (20, 100000),
            'brooches': (10, 10000)
        }

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            for category, (min_price, max_price) in price_ranges.items():
                cursor.execute("""
                    SELECT listing_id, price, category
                    FROM jewelry_listings
                    WHERE category = ? AND price IS NOT NULL 
                    AND (price < ? OR price > ?)
                """, (category, min_price, max_price))

                for row in cursor.fetchall():
                    severity = ValidationSeverity.WARNING if row[1] > max_price * \
                        2 else ValidationSeverity.INFO

                    issues.append(ValidationIssue(
                        rule_id="business_price_reasonable",
                        severity=severity,
                        category=ValidationCategory.BUSINESS_RULES,
                        table="jewelry_listings",
                        field_name="price",
                        record_id=row[0],
                        message=f"Price ${row[1]} may be unusual for {category} (typical range: ${min_price}-${max_price})",
                        current_value=row[1],
                        additional_info={'category': category,
                                         'typical_range': (min_price, max_price)}
                    ))

        return issues

    def _check_original_price_logic(self) -> List[ValidationIssue]:
        """Check original price business logic"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, price, original_price
                FROM jewelry_listings
                WHERE price IS NOT NULL AND original_price IS NOT NULL
                AND original_price < price
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="business_original_price_logic",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.BUSINESS_RULES,
                    table="jewelry_listings",
                    record_id=row[0],
                    message=f"Original price (${row[2]}) is less than current price (${row[1]})",
                    current_value={'price': row[1], 'original_price': row[2]},
                    additional_info={'price_difference': row[1] - row[2]}
                ))

        return issues

    def _check_images_quality_correlation(self) -> List[ValidationIssue]:
        """Check correlation between image count and quality score"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Find listings with high quality but no images
            cursor.execute("""
                SELECT listing_id, data_completeness_score, image_count
                FROM jewelry_listings
                WHERE data_completeness_score > 0.8 AND image_count = 0
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="business_images_quality_correlation",
                    severity=ValidationSeverity.INFO,
                    category=ValidationCategory.BUSINESS_RULES,
                    table="jewelry_listings",
                    record_id=row[0],
                    message=f"High quality score ({row[1]:.2f}) but no images",
                    current_value={
                        'quality_score': row[1], 'image_count': row[2]},
                    additional_info={
                        'recommendation': 'Consider adding images to improve listing'}
                ))

        return issues

    def _check_orphaned_images(self) -> List[ValidationIssue]:
        """Check for orphaned image records"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT image_id, listing_id
                FROM jewelry_images
                WHERE listing_id NOT IN (SELECT listing_id FROM jewelry_listings)
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="integrity_orphaned_images",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.REFERENTIAL_INTEGRITY,
                    table="jewelry_images",
                    record_id=row[0],
                    message=f"Image references non-existent listing: {row[1]}",
                    current_value=row[1],
                    additional_info={
                        'action': 'Delete orphaned image or restore listing'}
                ))

        return issues

    def _check_orphaned_specifications(self) -> List[ValidationIssue]:
        """Check for orphaned specification records"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT spec_id, listing_id
                FROM jewelry_specifications
                WHERE listing_id NOT IN (SELECT listing_id FROM jewelry_listings)
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="integrity_orphaned_specs",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.REFERENTIAL_INTEGRITY,
                    table="jewelry_specifications",
                    record_id=row[0],
                    message=f"Specification references non-existent listing: {row[1]}",
                    current_value=row[1],
                    additional_info={
                        'action': 'Delete orphaned specification or restore listing'}
                ))

        return issues

    def _check_title_quality(self) -> List[ValidationIssue]:
        """Check title quality and descriptiveness"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, title
                FROM jewelry_listings
                WHERE title IS NOT NULL AND title != ''
            """)

            for row in cursor.fetchall():
                title = row[1].strip()
                title_issues = []

                # Check length
                if len(title) < 10:
                    title_issues.append("too short")
                elif len(title) > 200:
                    title_issues.append("too long")

                # Check for all caps
                if title.isupper() and len(title) > 20:
                    title_issues.append("all caps")

                # Check for meaningful content
                if len(title.split()) < 3:
                    title_issues.append("too few words")

                # Check for common issues
                if title.lower().count('!!!') > 0:
                    title_issues.append("excessive exclamation marks")

                if title_issues:
                    severity = ValidationSeverity.WARNING if len(
                        title_issues) > 1 else ValidationSeverity.INFO

                    issues.append(ValidationIssue(
                        rule_id="quality_title_descriptive",
                        severity=severity,
                        category=ValidationCategory.DATA_QUALITY,
                        table="jewelry_listings",
                        field_name="title",
                        record_id=row[0],
                        message=f"Title quality issues: {', '.join(title_issues)}",
                        current_value=title,
                        additional_info={'issues': title_issues}
                    ))

        return issues

    def _check_description_completeness(self) -> List[ValidationIssue]:
        """Check description completeness and quality"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, description, description_length
                FROM jewelry_listings
                WHERE description IS NOT NULL AND description != ''
            """)

            for row in cursor.fetchall():
                description = row[1]
                desc_length = len(description) if description else 0

                if desc_length < 50:
                    issues.append(ValidationIssue(
                        rule_id="quality_description_completeness",
                        severity=ValidationSeverity.INFO,
                        category=ValidationCategory.DATA_QUALITY,
                        table="jewelry_listings",
                        field_name="description",
                        record_id=row[0],
                        message=f"Description is very short ({desc_length} characters)",
                        current_value=desc_length,
                        additional_info={
                            'recommendation': 'Add more descriptive details'}
                    ))
                elif desc_length > 5000:
                    issues.append(ValidationIssue(
                        rule_id="quality_description_completeness",
                        severity=ValidationSeverity.INFO,
                        category=ValidationCategory.DATA_QUALITY,
                        table="jewelry_listings",
                        field_name="description",
                        record_id=row[0],
                        message=f"Description is very long ({desc_length} characters)",
                        current_value=desc_length,
                        additional_info={
                            'recommendation': 'Consider condensing key information'}
                    ))

        return issues

    def _check_duplicate_listings(self) -> List[ValidationIssue]:
        """Check for potential duplicate listings"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Find listings with identical URLs
            cursor.execute("""
                SELECT url, GROUP_CONCAT(listing_id) as ids, COUNT(*) as count
                FROM jewelry_listings
                WHERE url IS NOT NULL AND url != ''
                GROUP BY url
                HAVING COUNT(*) > 1
            """)

            for row in cursor.fetchall():
                listing_ids = row[1].split(',')
                for listing_id in listing_ids[1:]:  # Skip first one
                    issues.append(ValidationIssue(
                        rule_id="quality_duplicate_listings",
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.DATA_QUALITY,
                        table="jewelry_listings",
                        record_id=listing_id,
                        message=f"Potential duplicate listing (same URL as {listing_ids[0]})",
                        current_value=row[0],
                        additional_info={
                            'duplicate_of': listing_ids[0], 'total_duplicates': row[2]}
                    ))

        return issues

    def _check_category_material_consistency(self) -> List[ValidationIssue]:
        """Check category-material combinations for consistency"""
        issues = []

        # Define unusual combinations
        unusual_combinations = {
            'watches': ['fabric', 'wood', 'plastic'],
            'rings': ['fabric', 'leather'],
            'earrings': ['leather', 'fabric'],
            'necklaces': ['plastic'],
            'bracelets': []
        }

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            for category, unusual_materials in unusual_combinations.items():
                if unusual_materials:
                    placeholders = ', '.join(['?' for _ in unusual_materials])
                    cursor.execute(f"""
                        SELECT listing_id, category, material
                        FROM jewelry_listings
                        WHERE category = ? AND material IN ({placeholders})
                    """, [category] + unusual_materials)

                    for row in cursor.fetchall():
                        issues.append(ValidationIssue(
                            rule_id="consistency_category_material",
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.CONSISTENCY,
                            table="jewelry_listings",
                            record_id=row[0],
                            message=f"Unusual material '{row[2]}' for category '{row[1]}'",
                            current_value={
                                'category': row[1], 'material': row[2]},
                            additional_info={'review_required': True}
                        ))

        return issues

    def _check_price_category_consistency(self) -> List[ValidationIssue]:
        """Check price consistency within categories"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Get price statistics by category
            cursor.execute("""
                SELECT 
                    category,
                    AVG(price) as avg_price,
                    AVG(price) + 3 * SQRT(AVG((price - (SELECT AVG(price) FROM jewelry_listings sub WHERE sub.category = jewelry_listings.category)) * 
                                                (price - (SELECT AVG(price) FROM jewelry_listings sub WHERE sub.category = jewelry_listings.category)))) as upper_threshold
                FROM jewelry_listings
                WHERE price > 0
                GROUP BY category
                HAVING COUNT(*) >= 10
            """)

            price_thresholds = {row[0]: row[2] for row in cursor.fetchall()}

            # Check for outliers
            for category, threshold in price_thresholds.items():
                cursor.execute("""
                    SELECT listing_id, price, category
                    FROM jewelry_listings
                    WHERE category = ? AND price > ?
                """, (category, threshold))

                for row in cursor.fetchall():
                    issues.append(ValidationIssue(
                        rule_id="consistency_price_category",
                        severity=ValidationSeverity.INFO,
                        category=ValidationCategory.CONSISTENCY,
                        table="jewelry_listings",
                        field_name="price",
                        record_id=row[0],
                        message=f"Price ${row[1]} is unusually high for {category} category",
                        current_value=row[1],
                        additional_info={'category': category,
                                         'threshold': threshold}
                    ))

        return issues

    def _check_essential_fields_completeness(self) -> List[ValidationIssue]:
        """Check completeness of essential fields"""
        issues = []

        essential_fields = ['brand', 'main_stone',
                            'size', 'weight', 'description']

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            for field in essential_fields:
                try:
                    cursor.execute(f"""
                        SELECT listing_id, {field}
                        FROM jewelry_listings
                        WHERE {field} IS NULL OR {field} = ''
                    """)

                    count = 0
                    for row in cursor.fetchall():
                        count += 1
                        if count <= 10:  # Limit to first 10 for performance
                            issues.append(ValidationIssue(
                                rule_id="completeness_essential_fields",
                                severity=ValidationSeverity.WARNING,
                                category=ValidationCategory.COMPLETENESS,
                                table="jewelry_listings",
                                field_name=field,
                                record_id=row[0],
                                message=f"Essential field '{field}' is missing",
                                current_value=row[1],
                                additional_info={
                                    'recommendation': f'Add {field} information'}
                            ))

                except sqlite3.OperationalError:
                    # Field doesn't exist, skip
                    pass

        return issues

    def _check_images_available(self) -> List[ValidationIssue]:
        """Check if listings have images"""
        issues = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT listing_id, image_count
                FROM jewelry_listings
                WHERE image_count = 0 OR image_count IS NULL
                LIMIT 50
            """)

            for row in cursor.fetchall():
                issues.append(ValidationIssue(
                    rule_id="completeness_images_available",
                    severity=ValidationSeverity.INFO,
                    category=ValidationCategory.COMPLETENESS,
                    table="jewelry_listings",
                    field_name="image_count",
                    record_id=row[0],
                    message="Listing has no images",
                    current_value=row[1] or 0,
                    additional_info={
                        'recommendation': 'Add product images to improve listing quality'}
                ))

        return issues

    # === MAIN VALIDATION METHODS ===

    def validate_database(self, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """
        Perform comprehensive database validation

        Args:
            level: Validation strictness level

        Returns:
            ValidationReport with findings and recommendations
        """

        validation_id = self._generate_validation_id()
        report = ValidationReport(
            validation_id=validation_id,
            level=level,
            started_at=datetime.now()
        )

        start_time = datetime.now()

        try:
            logger.info(f"Starting database validation with level: {level}")

            # Filter rules by validation level
            applicable_rules = [
                rule for rule in self.validation_rules
                if self._should_apply_rule(rule, level)
            ]

            report.rules_executed = len(applicable_rules)

            # Execute validation rules
            for rule in applicable_rules:
                try:
                    rule_issues = rule.check_function()
                    for issue in rule_issues:
                        report.add_issue(issue)

                except Exception as e:
                    logger.error(f"Validation rule {rule.rule_id} failed: {e}")
                    # Add error as critical issue
                    report.add_issue(ValidationIssue(
                        rule_id=rule.rule_id,
                        severity=ValidationSeverity.CRITICAL,
                        category=ValidationCategory.DATA_QUALITY,
                        table=rule.table,
                        field=rule.field_name,
                        record_id=None,
                        message=f"Validation rule execution failed: {e}",
                        additional_info={'rule_name': rule.name}
                    ))

            # Calculate quality scores
            report.overall_quality_score = self._calculate_overall_quality_score(
                report)
            report.table_quality_scores = self._calculate_table_quality_scores(
                report)

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            # Get record counts
            report.total_records_checked = self._get_total_record_count()

            # Finalize report
            report.completed_at = datetime.now()
            report.execution_time_seconds = (
                report.completed_at - start_time).total_seconds()

            logger.info(
                f"Validation completed in {report.execution_time_seconds:.2f}s. Found {report.total_issues_found} issues.")

            return report

        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            report.completed_at = datetime.now()
            report.execution_time_seconds = (
                report.completed_at - start_time).total_seconds()

            # Add critical failure issue
            report.add_issue(ValidationIssue(
                rule_id="validation_engine_failure",
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.DATA_QUALITY,
                table=None,
                field=None,
                record_id=None,
                message=f"Validation engine failure: {e}"
            ))

            return report

    def validate_record(self, table: str, record_id: str, level: ValidationLevel = ValidationLevel.STANDARD) -> List[ValidationIssue]:
        """
        Validate a specific record

        Args:
            table: Table name
            record_id: Record identifier
            level: Validation level

        Returns:
            List of validation issues for the record
        """

        issues = []

        # Filter rules for the specific table
        table_rules = [
            rule for rule in self.validation_rules
            if (rule.table == table or rule.table is None) and self._should_apply_rule(rule, level)
        ]

        # This would need table-specific implementations
        # For now, return empty list
        return issues

    def _should_apply_rule(self, rule: ValidationRule, level: ValidationLevel) -> bool:
        """Determine if a rule should be applied for the given validation level"""

        level_hierarchy = {
            ValidationLevel.BASIC: 1,
            ValidationLevel.STANDARD: 2,
            ValidationLevel.STRICT: 3,
            ValidationLevel.PEDANTIC: 4
        }

        return level_hierarchy[level] >= level_hierarchy[rule.required_level]

    def _calculate_overall_quality_score(self, report: ValidationReport) -> float:
        """Calculate overall database quality score"""

        if report.total_records_checked == 0:
            return 0.0

        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.8,
            ValidationSeverity.WARNING: 0.5,
            ValidationSeverity.INFO: 0.2
        }

        total_weight = sum(
            report.issues_by_severity.get(severity, 0) * weight
            for severity, weight in severity_weights.items()
        )

        # Calculate score (1.0 = perfect, 0.0 = terrible)
        max_possible_issues = report.total_records_checked * \
            0.1  # Assume 10% issues is very bad
        quality_score = max(0.0, 1.0 - (total_weight / max_possible_issues))

        return round(quality_score, 3)

    def _calculate_table_quality_scores(self, report: ValidationReport) -> Dict[str, float]:
        """Calculate quality scores by table"""

        table_scores = {}

        # Group issues by table
        issues_by_table = {}
        for issue in report.issues:
            if issue.table:
                if issue.table not in issues_by_table:
                    issues_by_table[issue.table] = []
                issues_by_table[issue.table].append(issue)

        # Calculate score for each table
        for table, issues in issues_by_table.items():
            severity_weights = {
                ValidationSeverity.CRITICAL: 1.0,
                ValidationSeverity.ERROR: 0.8,
                ValidationSeverity.WARNING: 0.5,
                ValidationSeverity.INFO: 0.2
            }

            total_weight = sum(severity_weights.get(
                issue.severity, 0.2) for issue in issues)

            # Get record count for table
            table_record_count = self._get_table_record_count(table)

            if table_record_count > 0:
                max_issues = table_record_count * 0.1
                score = max(0.0, 1.0 - (total_weight / max_issues))
                table_scores[table] = round(score, 3)

        return table_scores

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []

        # Critical issues
        critical_count = report.issues_by_severity.get(
            ValidationSeverity.CRITICAL, 0)
        if critical_count > 0:
            recommendations.append(
                f"Address {critical_count} critical issues immediately")

        # Error issues
        error_count = report.issues_by_severity.get(
            ValidationSeverity.ERROR, 0)
        if error_count > 0:
            recommendations.append(
                f"Fix {error_count} data errors to improve reliability")

        # Warning patterns
        warning_count = report.issues_by_severity.get(
            ValidationSeverity.WARNING, 0)
        if warning_count > 50:
            recommendations.append(
                "High number of warnings suggests systematic data quality issues")

        # Category-specific recommendations
        category_issues = report.issues_by_category
        if category_issues.get(ValidationCategory.REQUIRED_FIELDS, 0) > 10:
            recommendations.append(
                "Improve data collection to ensure required fields are populated")

        if category_issues.get(ValidationCategory.FORMAT_VALIDATION, 0) > 20:
            recommendations.append(
                "Implement input validation to prevent format issues")

        if category_issues.get(ValidationCategory.COMPLETENESS, 0) > 100:
            recommendations.append(
                "Focus on data completeness - many fields are missing")

        # Quality score recommendations
        if report.overall_quality_score and report.overall_quality_score < 0.7:
            recommendations.append(
                "Overall data quality is below recommended threshold (70%)")

        return recommendations

    def _get_total_record_count(self) -> int:
        """Get total number of records across all tables"""

        try:
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM jewelry_listings")
                return cursor.fetchone()[0]

        except Exception:
            return 0

    def _get_table_record_count(self, table: str) -> int:
        """Get record count for specific table"""

        try:
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                return cursor.fetchone()[0]

        except Exception:
            return 0

    def _generate_validation_id(self) -> str:
        """Generate unique validation ID"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"validation_{timestamp}_{hash(datetime.now()) % 10000:04d}"

    def get_validation_summary(self, report: ValidationReport) -> Dict[str, Any]:
        """Get a summary of validation results"""

        return {
            'validation_id': report.validation_id,
            'level': report.level.value,
            'execution_time_seconds': report.execution_time_seconds,
            'total_records_checked': report.total_records_checked,
            'total_issues_found': report.total_issues_found,
            'overall_quality_score': report.overall_quality_score,
            'issues_by_severity': {k.value: v for k, v in report.issues_by_severity.items()},
            'issues_by_category': {k.value: v for k, v in report.issues_by_category.items()},
            'top_recommendations': report.recommendations[:5],
            'tables_analyzed': list(report.table_quality_scores.keys()),
            'completion_status': 'completed' if report.completed_at else 'running'
        }


class IntegrityChecker:
    """
    Specialized integrity checker for database consistency
    """

    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager

    def check_foreign_key_constraints(self) -> List[Dict[str, Any]]:
        """Check foreign key constraint violations"""

        violations = []

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Enable foreign key checking
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA foreign_key_check")

            for row in cursor.fetchall():
                violations.append({
                    'table': row[0],
                    'rowid': row[1],
                    'parent_table': row[2],
                    'fk_index': row[3]
                })

        return violations

    def check_database_integrity(self) -> Dict[str, Any]:
        """Perform SQLite integrity check"""

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]

            return {
                'status': 'ok' if result == 'ok' else 'error',
                'details': result,
                'checked_at': datetime.now().isoformat()
            }

    def analyze_table_statistics(self) -> Dict[str, Any]:
        """Analyze table statistics for anomalies"""

        stats = {}
        tables = ['jewelry_listings', 'jewelry_images',
                  'jewelry_specifications', 'scraping_sessions']

        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]

                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()

                    stats[table] = {
                        'row_count': row_count,
                        'column_count': len(columns),
                        'columns': [col[1] for col in columns]
                    }

                except sqlite3.OperationalError as e:
                    stats[table] = {'error': str(e)}

        return stats


# Convenience functions for common validation scenarios
def quick_validation(db_manager: DatabaseManager) -> ValidationReport:
    """Perform quick basic validation"""

    validator = ValidationEngine(db_manager)
    return validator.validate_database(ValidationLevel.BASIC)


def comprehensive_validation(db_manager: DatabaseManager) -> ValidationReport:
    """Perform comprehensive validation"""

    validator = ValidationEngine(db_manager)
    return validator.validate_database(ValidationLevel.STRICT)


def integrity_check(db_manager: DatabaseManager) -> Dict[str, Any]:
    """Perform database integrity check"""

    checker = IntegrityChecker(db_manager)

    return {
        'foreign_key_violations': checker.check_foreign_key_constraints(),
        'database_integrity': checker.check_database_integrity(),
        'table_statistics': checker.analyze_table_statistics()
    }
