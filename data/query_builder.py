"""
Advanced Query Builder for Jewelry Database
Provides sophisticated query construction with fluent interface and type safety.
"""

import re
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .models import JewelryCategory, JewelryMaterial, ListingStatus, DataQuality


class SortOrder(str, Enum):
    """Sort order enumeration"""
    ASC = "ASC"
    DESC = "DESC"


class ComparisonOperator(str, Enum):
    """Comparison operators for filters"""
    EQ = "="           # Equal
    NE = "!="          # Not equal
    GT = ">"           # Greater than
    GTE = ">="         # Greater than or equal
    LT = "<"           # Less than
    LTE = "<="         # Less than or equal
    LIKE = "LIKE"      # Pattern matching
    ILIKE = "ILIKE"    # Case-insensitive pattern matching
    IN = "IN"          # In list
    NOT_IN = "NOT IN"  # Not in list
    BETWEEN = "BETWEEN" # Between values
    IS_NULL = "IS NULL"     # Is null
    IS_NOT_NULL = "IS NOT NULL"  # Is not null


class JoinType(str, Enum):
    """SQL join types"""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"


@dataclass
class FilterCondition:
    """Represents a single filter condition"""
    field: str
    operator: ComparisonOperator
    value: Any = None
    value2: Any = None  # For BETWEEN operator
    table_alias: Optional[str] = None
    
    def to_sql(self) -> Tuple[str, List[Any]]:
        """Convert filter condition to SQL and parameters"""
        field_name = f"{self.table_alias}.{self.field}" if self.table_alias else self.field
        params = []
        
        if self.operator == ComparisonOperator.IS_NULL:
            return f"{field_name} IS NULL", params
        elif self.operator == ComparisonOperator.IS_NOT_NULL:
            return f"{field_name} IS NOT NULL", params
        elif self.operator == ComparisonOperator.BETWEEN:
            params = [self.value, self.value2]
            return f"{field_name} BETWEEN ? AND ?", params
        elif self.operator == ComparisonOperator.IN:
            if isinstance(self.value, (list, tuple)):
                placeholders = ", ".join(["?" for _ in self.value])
                params = list(self.value)
                return f"{field_name} IN ({placeholders})", params
            else:
                params = [self.value]
                return f"{field_name} IN (?)", params
        elif self.operator == ComparisonOperator.NOT_IN:
            if isinstance(self.value, (list, tuple)):
                placeholders = ", ".join(["?" for _ in self.value])
                params = list(self.value)
                return f"{field_name} NOT IN ({placeholders})", params
            else:
                params = [self.value]
                return f"{field_name} NOT IN (?)", params
        elif self.operator == ComparisonOperator.LIKE:
            params = [self.value]
            return f"{field_name} LIKE ?", params
        elif self.operator == ComparisonOperator.ILIKE:
            # SQLite doesn't have ILIKE, use LIKE with UPPER
            params = [str(self.value).upper()]
            return f"UPPER({field_name}) LIKE ?", params
        else:
            params = [self.value]
            return f"{field_name} {self.operator.value} ?", params


@dataclass
class SortOption:
    """Represents a sort option"""
    field: str
    order: SortOrder = SortOrder.ASC
    table_alias: Optional[str] = None
    
    def to_sql(self) -> str:
        """Convert sort option to SQL"""
        field_name = f"{self.table_alias}.{self.field}" if self.table_alias else self.field
        return f"{field_name} {self.order.value}"


@dataclass
class JoinClause:
    """Represents a join clause"""
    join_type: JoinType
    table: str
    alias: str
    on_condition: str
    
    def to_sql(self) -> str:
        """Convert join clause to SQL"""
        return f"{self.join_type.value} {self.table} {self.alias} ON {self.on_condition}"


@dataclass
class QueryFilters:
    """Enhanced query filters with comprehensive options"""
    
    # Basic filters
    listing_ids: Optional[List[str]] = None
    category: Optional[Union[str, JewelryCategory, List[Union[str, JewelryCategory]]]] = None
    material: Optional[Union[str, JewelryMaterial, List[Union[str, JewelryMaterial]]]] = None
    brand: Optional[Union[str, List[str]]] = None
    seller: Optional[Union[str, List[str]]] = None
    condition: Optional[Union[str, List[str]]] = None
    listing_status: Optional[Union[str, ListingStatus, List[Union[str, ListingStatus]]]] = None
    
    # Price filters
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: Optional[Union[str, List[str]]] = None
    has_original_price: Optional[bool] = None
    price_drop_min: Optional[float] = None
    
    # Quality filters
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    quality_class: Optional[Union[DataQuality, List[DataQuality]]] = None
    is_validated: Optional[bool] = None
    min_validation_score: Optional[float] = None
    
    # Date filters
    scraped_after: Optional[datetime] = None
    scraped_before: Optional[datetime] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    listing_date_after: Optional[datetime] = None
    listing_date_before: Optional[datetime] = None
    
    # Engagement filters
    min_watchers: Optional[int] = None
    max_watchers: Optional[int] = None
    min_views: Optional[int] = None
    max_views: Optional[int] = None
    min_bids: Optional[int] = None
    has_watchers: Optional[bool] = None
    has_bids: Optional[bool] = None
    
    # Seller filters
    min_seller_rating: Optional[float] = None
    max_seller_rating: Optional[float] = None
    min_feedback_count: Optional[int] = None
    seller_level: Optional[Union[str, List[str]]] = None
    
    # Image filters
    has_images: Optional[bool] = None
    min_image_count: Optional[int] = None
    max_image_count: Optional[int] = None
    min_image_quality: Optional[float] = None
    has_main_image: Optional[bool] = None
    
    # Content filters
    search_text: Optional[str] = None
    search_title: Optional[str] = None
    search_description: Optional[str] = None
    min_description_length: Optional[int] = None
    has_features: Optional[bool] = None
    has_specifications: Optional[bool] = None
    
    # Jewelry-specific filters
    gemstone: Optional[Union[str, List[str]]] = None
    has_gemstone: Optional[bool] = None
    stone_color: Optional[Union[str, List[str]]] = None
    stone_clarity: Optional[Union[str, List[str]]] = None
    stone_cut: Optional[Union[str, List[str]]] = None
    size: Optional[Union[str, List[str]]] = None
    era: Optional[Union[str, List[str]]] = None
    designer: Optional[Union[str, List[str]]] = None
    collection: Optional[Union[str, List[str]]] = None
    
    # Shipping filters
    free_shipping: Optional[bool] = None
    international_shipping: Optional[bool] = None
    expedited_shipping: Optional[bool] = None
    ships_from: Optional[Union[str, List[str]]] = None
    
    # Advanced filters
    has_video: Optional[bool] = None
    best_offer_available: Optional[bool] = None
    auction_format: Optional[bool] = None
    buy_it_now_format: Optional[bool] = None
    
    # Custom filters
    custom_conditions: List[FilterCondition] = field(default_factory=list)


@dataclass
class SortOptions:
    """Sort options for query results"""
    
    # Predefined sort options
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.ASC
    
    # Multiple sort fields
    sort_fields: List[SortOption] = field(default_factory=list)
    
    # Common sort presets
    sort_by_price_asc: bool = False
    sort_by_price_desc: bool = False
    sort_by_quality_desc: bool = False
    sort_by_popularity_desc: bool = False
    sort_by_newest: bool = False
    sort_by_ending_soon: bool = False


class QueryBuilder:
    """
    Advanced query builder for jewelry database with fluent interface
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the query builder to initial state"""
        self._select_fields: List[str] = []
        self._from_table: str = "jewelry_listings"
        self._table_alias: str = "l"
        self._joins: List[JoinClause] = []
        self._where_conditions: List[FilterCondition] = []
        self._group_by: List[str] = []
        self._having_conditions: List[FilterCondition] = []
        self._sort_options: List[SortOption] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._distinct: bool = False
        
        return self
    
    # === SELECT METHODS ===
    
    def select(self, *fields: str):
        """Add fields to SELECT clause"""
        self._select_fields.extend(fields)
        return self
    
    def select_all(self):
        """Select all fields"""
        self._select_fields = ["*"]
        return self
    
    def select_summary(self):
        """Select common summary fields"""
        summary_fields = [
            f"{self._table_alias}.listing_id",
            f"{self._table_alias}.title",
            f"{self._table_alias}.price",
            f"{self._table_alias}.currency",
            f"{self._table_alias}.category",
            f"{self._table_alias}.material",
            f"{self._table_alias}.brand",
            f"{self._table_alias}.seller_name",
            f"{self._table_alias}.condition",
            f"{self._table_alias}.data_completeness_score",
            f"{self._table_alias}.image_count",
            f"{self._table_alias}.watchers",
            f"{self._table_alias}.views",
            f"{self._table_alias}.scraped_at",
            f"{self._table_alias}.url"
        ]
        self._select_fields.extend(summary_fields)
        return self
    
    def distinct(self):
        """Add DISTINCT to query"""
        self._distinct = True
        return self
    
    # === FROM AND JOIN METHODS ===
    
    def from_table(self, table: str, alias: str = None):
        """Set the FROM table"""
        self._from_table = table
        if alias:
            self._table_alias = alias
        return self
    
    def join_images(self, join_type: JoinType = JoinType.LEFT):
        """Join with images table"""
        join_clause = JoinClause(
            join_type=join_type,
            table="jewelry_images",
            alias="img",
            on_condition=f"{self._table_alias}.listing_id = img.listing_id"
        )
        self._joins.append(join_clause)
        return self
    
    def join_specifications(self, join_type: JoinType = JoinType.LEFT):
        """Join with specifications table"""
        join_clause = JoinClause(
            join_type=join_type,
            table="jewelry_specifications",
            alias="spec",
            on_condition=f"{self._table_alias}.listing_id = spec.listing_id"
        )
        self._joins.append(join_clause)
        return self
    
    def join_sessions(self, join_type: JoinType = JoinType.LEFT):
        """Join with scraping sessions table"""
        join_clause = JoinClause(
            join_type=join_type,
            table="scraping_sessions",
            alias="sess",
            on_condition=f"{self._table_alias}.scraped_at BETWEEN sess.started_at AND COALESCE(sess.completed_at, datetime('now'))"
        )
        self._joins.append(join_clause)
        return self
    
    # === FILTER METHODS ===
    
    def where(self, field: str, operator: ComparisonOperator, value: Any, value2: Any = None):
        """Add a WHERE condition"""
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value,
            value2=value2,
            table_alias=self._table_alias
        )
        self._where_conditions.append(condition)
        return self
    
    def category(self, category: Union[str, JewelryCategory, List[Union[str, JewelryCategory]]]):
        """Filter by category"""
        if isinstance(category, list):
            categories = [c.value if isinstance(c, JewelryCategory) else c for c in category]
            return self.where("category", ComparisonOperator.IN, categories)
        else:
            cat_value = category.value if isinstance(category, JewelryCategory) else category
            return self.where("category", ComparisonOperator.EQ, cat_value)
    
    def material(self, material: Union[str, JewelryMaterial, List[Union[str, JewelryMaterial]]]):
        """Filter by material"""
        if isinstance(material, list):
            materials = [m.value if isinstance(m, JewelryMaterial) else m for m in material]
            return self.where("material", ComparisonOperator.IN, materials)
        else:
            mat_value = material.value if isinstance(material, JewelryMaterial) else material
            return self.where("material", ComparisonOperator.EQ, mat_value)
    
    def price_range(self, min_price: float = None, max_price: float = None):
        """Filter by price range"""
        if min_price is not None:
            self.where("price", ComparisonOperator.GTE, min_price)
        if max_price is not None:
            self.where("price", ComparisonOperator.LTE, max_price)
        return self
    
    def price_between(self, min_price: float, max_price: float):
        """Filter by price between values"""
        return self.where("price", ComparisonOperator.BETWEEN, min_price, max_price)
    
    def brand(self, brand: Union[str, List[str]]):
        """Filter by brand"""
        if isinstance(brand, list):
            return self.where("brand", ComparisonOperator.IN, brand)
        else:
            return self.where("brand", ComparisonOperator.EQ, brand)
    
    def seller(self, seller: Union[str, List[str]]):
        """Filter by seller"""
        if isinstance(seller, list):
            return self.where("seller_name", ComparisonOperator.IN, seller)
        else:
            return self.where("seller_name", ComparisonOperator.EQ, seller)
    
    def quality_score_range(self, min_score: float = None, max_score: float = None):
        """Filter by quality score range"""
        if min_score is not None:
            self.where("data_completeness_score", ComparisonOperator.GTE, min_score)
        if max_score is not None:
            self.where("data_completeness_score", ComparisonOperator.LTE, max_score)
        return self
    
    def high_quality(self, threshold: float = 0.8):
        """Filter for high quality listings"""
        return self.where("data_completeness_score", ComparisonOperator.GTE, threshold)
    
    def validated_only(self):
        """Filter for validated listings only"""
        return self.where("is_validated", ComparisonOperator.EQ, True)
    
    def has_images(self, min_count: int = 1):
        """Filter for listings with images"""
        return self.where("image_count", ComparisonOperator.GTE, min_count)
    
    def has_watchers(self, min_watchers: int = 1):
        """Filter for listings with watchers"""
        return self.where("watchers", ComparisonOperator.GTE, min_watchers)
    
    def has_bids(self, min_bids: int = 1):
        """Filter for listings with bids"""
        return self.where("bids", ComparisonOperator.GTE, min_bids)
    
    def scraped_after(self, date: datetime):
        """Filter for listings scraped after date"""
        return self.where("scraped_at", ComparisonOperator.GTE, date.isoformat())
    
    def scraped_before(self, date: datetime):
        """Filter for listings scraped before date"""
        return self.where("scraped_at", ComparisonOperator.LTE, date.isoformat())
    
    def scraped_today(self):
        """Filter for listings scraped today"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.scraped_after(today)
    
    def scraped_last_week(self):
        """Filter for listings scraped in the last week"""
        week_ago = datetime.now() - timedelta(days=7)
        return self.scraped_after(week_ago)
    
    def search_text(self, text: str, fields: List[str] = None):
        """Search text in specified fields"""
        if not fields:
            fields = ["title", "description"]
        
        search_pattern = f"%{text}%"
        
        # Create OR conditions for each field
        for i, field in enumerate(fields):
            if i == 0:
                self.where(field, ComparisonOperator.LIKE, search_pattern)
            else:
                # For additional fields, we need to manually add OR logic
                # This is a simplified version - for complex OR conditions,
                # consider using raw SQL or enhancing the query builder
                pass
        
        return self
    
    def gemstone(self, gemstone: Union[str, List[str]]):
        """Filter by gemstone"""
        if isinstance(gemstone, list):
            return self.where("main_stone", ComparisonOperator.IN, gemstone)
        else:
            return self.where("main_stone", ComparisonOperator.EQ, gemstone)
    
    def has_gemstone(self):
        """Filter for listings with gemstones"""
        return self.where("main_stone", ComparisonOperator.IS_NOT_NULL)
    
    # === SORTING METHODS ===
    
    def order_by(self, field: str, order: SortOrder = SortOrder.ASC, table_alias: str = None):
        """Add ORDER BY clause"""
        sort_option = SortOption(
            field=field,
            order=order,
            table_alias=table_alias or self._table_alias
        )
        self._sort_options.append(sort_option)
        return self
    
    def order_by_price(self, ascending: bool = True):
        """Order by price"""
        order = SortOrder.ASC if ascending else SortOrder.DESC
        return self.order_by("price", order)
    
    def order_by_quality(self, ascending: bool = False):
        """Order by quality score"""
        order = SortOrder.ASC if ascending else SortOrder.DESC
        return self.order_by("data_completeness_score", order)
    
    def order_by_popularity(self):
        """Order by popularity (watchers + views)"""
        # This would require a calculated field - simplified for now
        return self.order_by("watchers", SortOrder.DESC)
    
    def order_by_newest(self):
        """Order by newest scraped first"""
        return self.order_by("scraped_at", SortOrder.DESC)
    
    def order_by_ending_soon(self):
        """Order by ending soonest (for auctions)"""
        return self.order_by("end_time", SortOrder.ASC)
    
    # === AGGREGATION METHODS ===
    
    def group_by(self, *fields: str):
        """Add GROUP BY clause"""
        self._group_by.extend(fields)
        return self
    
    def having(self, field: str, operator: ComparisonOperator, value: Any):
        """Add HAVING condition"""
        condition = FilterCondition(
            field=field,
            operator=operator,
            value=value
        )
        self._having_conditions.append(condition)
        return self
    
    # === PAGINATION METHODS ===
    
    def limit(self, count: int):
        """Set LIMIT"""
        self._limit = count
        return self
    
    def offset(self, count: int):
        """Set OFFSET"""
        self._offset = count
        return self
    
    def paginate(self, page: int, page_size: int):
        """Set pagination"""
        self._limit = page_size
        self._offset = (page - 1) * page_size
        return self
    
    # === QUERY BUILDING ===
    
    def apply_filters(self, filters: QueryFilters):
        """Apply QueryFilters object to the query builder"""
        # Basic filters
        if filters.listing_ids:
            self.where("listing_id", ComparisonOperator.IN, filters.listing_ids)
        
        if filters.category:
            self.category(filters.category)
        
        if filters.material:
            self.material(filters.material)
        
        if filters.brand:
            self.brand(filters.brand)
        
        if filters.seller:
            self.seller(filters.seller)
        
        if filters.condition:
            if isinstance(filters.condition, list):
                self.where("condition", ComparisonOperator.IN, filters.condition)
            else:
                self.where("condition", ComparisonOperator.EQ, filters.condition)
        
        # Price filters
        if filters.min_price is not None or filters.max_price is not None:
            self.price_range(filters.min_price, filters.max_price)
        
        if filters.currency:
            if isinstance(filters.currency, list):
                self.where("currency", ComparisonOperator.IN, filters.currency)
            else:
                self.where("currency", ComparisonOperator.EQ, filters.currency)
        
        # Quality filters
        if filters.min_quality_score is not None or filters.max_quality_score is not None:
            self.quality_score_range(filters.min_quality_score, filters.max_quality_score)
        
        if filters.is_validated is not None:
            self.where("is_validated", ComparisonOperator.EQ, filters.is_validated)
        
        # Date filters
        if filters.scraped_after:
            self.scraped_after(filters.scraped_after)
        
        if filters.scraped_before:
            self.scraped_before(filters.scraped_before)
        
        if filters.created_after:
            self.where("created_at", ComparisonOperator.GTE, filters.created_after.isoformat())
        
        if filters.created_before:
            self.where("created_at", ComparisonOperator.LTE, filters.created_before.isoformat())
        
        # Engagement filters
        if filters.min_watchers is not None:
            self.where("watchers", ComparisonOperator.GTE, filters.min_watchers)
        
        if filters.max_watchers is not None:
            self.where("watchers", ComparisonOperator.LTE, filters.max_watchers)
        
        if filters.min_views is not None:
            self.where("views", ComparisonOperator.GTE, filters.min_views)
        
        if filters.has_watchers is not None:
            if filters.has_watchers:
                self.where("watchers", ComparisonOperator.GT, 0)
            else:
                self.where("watchers", ComparisonOperator.EQ, 0)
        
        # Image filters
        if filters.has_images is not None:
            if filters.has_images:
                self.has_images()
            else:
                self.where("image_count", ComparisonOperator.EQ, 0)
        
        if filters.min_image_count is not None:
            self.where("image_count", ComparisonOperator.GTE, filters.min_image_count)
        
        # Search filters
        if filters.search_text:
            self.search_text(filters.search_text)
        
        # Jewelry-specific filters
        if filters.gemstone:
            self.gemstone(filters.gemstone)
        
        if filters.has_gemstone is not None:
            if filters.has_gemstone:
                self.has_gemstone()
            else:
                self.where("main_stone", ComparisonOperator.IS_NULL)
        
        # Custom conditions
        if filters.custom_conditions:
            self._where_conditions.extend(filters.custom_conditions)
        
        return self
    
    def apply_sort_options(self, sort_options: SortOptions):
        """Apply SortOptions object to the query builder"""
        # Clear existing sort options
        self._sort_options = []
        
        # Apply predefined sorts
        if sort_options.sort_by_price_asc:
            self.order_by_price(ascending=True)
        elif sort_options.sort_by_price_desc:
            self.order_by_price(ascending=False)
        elif sort_options.sort_by_quality_desc:
            self.order_by_quality(ascending=False)
        elif sort_options.sort_by_popularity_desc:
            self.order_by_popularity()
        elif sort_options.sort_by_newest:
            self.order_by_newest()
        elif sort_options.sort_by_ending_soon:
            self.order_by_ending_soon()
        elif sort_options.sort_by:
            self.order_by(sort_options.sort_by, sort_options.sort_order)
        
        # Apply custom sort fields
        if sort_options.sort_fields:
            self._sort_options.extend(sort_options.sort_fields)
        
        return self
    
    def build(self) -> Tuple[str, List[Any]]:
        """Build the final SQL query and parameters"""
        query_parts = []
        params = []
        
        # SELECT clause
        if self._select_fields:
            select_clause = "SELECT"
            if self._distinct:
                select_clause += " DISTINCT"
            select_clause += " " + ", ".join(self._select_fields)
        else:
            select_clause = f"SELECT {self._table_alias}.*"
        
        query_parts.append(select_clause)
        
        # FROM clause
        from_clause = f"FROM {self._from_table} {self._table_alias}"
        query_parts.append(from_clause)
        
        # JOIN clauses
        for join in self._joins:
            query_parts.append(join.to_sql())
        
        # WHERE clause
        if self._where_conditions:
            where_parts = []
            for condition in self._where_conditions:
                condition_sql, condition_params = condition.to_sql()
                where_parts.append(condition_sql)
                params.extend(condition_params)
            
            query_parts.append("WHERE " + " AND ".join(where_parts))
        
        # GROUP BY clause
        if self._group_by:
            query_parts.append("GROUP BY " + ", ".join(self._group_by))
        
        # HAVING clause
        if self._having_conditions:
            having_parts = []
            for condition in self._having_conditions:
                condition_sql, condition_params = condition.to_sql()
                having_parts.append(condition_sql)
                params.extend(condition_params)
            
            query_parts.append("HAVING " + " AND ".join(having_parts))
        
        # ORDER BY clause
        if self._sort_options:
            order_parts = [sort_option.to_sql() for sort_option in self._sort_options]
            query_parts.append("ORDER BY " + ", ".join(order_parts))
        
        # LIMIT clause
        if self._limit is not None:
            query_parts.append(f"LIMIT {self._limit}")
        
        # OFFSET clause
        if self._offset is not None:
            query_parts.append(f"OFFSET {self._offset}")
        
        final_query = " ".join(query_parts)
        return final_query, params
    
    def build_count_query(self) -> Tuple[str, List[Any]]:
        """Build a COUNT query for the same conditions"""
        # Save current state
        original_select = self._select_fields.copy()
        original_sort = self._sort_options.copy()
        original_limit = self._limit
        original_offset = self._offset
        original_distinct = self._distinct
        
        # Modify for count
        self._select_fields = ["COUNT(*) as total_count"]
        self._sort_options = []
        self._limit = None
        self._offset = None
        self._distinct = False
        
        # Build count query
        count_query, params = self.build()
        
        # Restore original state
        self._select_fields = original_select
        self._sort_options = original_sort
        self._limit = original_limit
        self._offset = original_offset
        self._distinct = original_distinct
        
        return count_query, params


# Convenience functions for common query patterns
def create_listing_query() -> QueryBuilder:
    """Create a query builder for listings with common setup"""
    return QueryBuilder().from_table("jewelry_listings", "l").select_summary()

def create_high_quality_query(threshold: float = 0.8) -> QueryBuilder:
    """Create a query for high-quality listings"""
    return (create_listing_query()
            .high_quality(threshold)
            .validated_only()
            .has_images()
            .order_by_quality())

def create_recent_listings_query(days: int = 7) -> QueryBuilder:
    """Create a query for recent listings"""
    cutoff_date = datetime.now() - timedelta(days=days)
    return (create_listing_query()
            .scraped_after(cutoff_date)
            .order_by_newest())

def create_popular_listings_query() -> QueryBuilder:
    """Create a query for popular listings"""
    return (create_listing_query()
            .has_watchers()
            .order_by_popularity())

def create_category_analysis_query(category: Union[str, JewelryCategory]) -> QueryBuilder:
    """Create a query for category analysis"""
    return (QueryBuilder()
            .select("category", "COUNT(*) as listing_count", 
                   "AVG(price) as avg_price", "MIN(price) as min_price", 
                   "MAX(price) as max_price", "AVG(data_completeness_score) as avg_quality")
            .from_table("jewelry_listings", "l")
            .category(category)
            .validated_only()
            .group_by("category"))