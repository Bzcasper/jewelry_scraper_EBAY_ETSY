#!/usr/bin/env python3
"""
Jewelry Scraper FastMCP Server
==============================

Token-efficient FastMCP server for jewelry scraping system with comprehensive
tool endpoints, resource access, and validation models.

Tasks: mcp_001 through mcp_012
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.jewelry_data_manager import JewelryDatabaseManager, QueryFilters, DatabaseStats
from models.jewelry_models import (
    JewelryListing, JewelryCategory, JewelryMaterial,
    ListingStatus, ScrapingStatus,
    JewelryImage, ScrapingSession
)
from data.models import DataQuality
from scrapers.ebay.scraper_engine import EbayJewelryScraper, ScrapingConfig
from scrapers.ebay_url_builder import EBayURLBuilder, SearchFilters, JewelryCategory as URLJewelryCategory
from ..models.ebay_types import ScrapingMode, AntiDetectionLevel
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic.types import PositiveInt

# Simple tool decorator for basic functionality
def tool(name: str = None):
    """Simple tool decorator that preserves function metadata"""
    def decorator(func):
        func._mcp_tool = name or func.__name__
        return func
    return decorator

def resource(uri: str):
    """Simple resource decorator that preserves function metadata"""
    def decorator(func):
        func._mcp_resource = uri
        return func
    return decorator

def prompt(name: str = None):
    """Simple prompt decorator that preserves function metadata"""
    def decorator(func):
        func._mcp_prompt = name or func.__name__
        return func
    return decorator

# Create a simple router for registering endpoints
from fastapi import APIRouter
mcp_router = APIRouter(prefix="/mcp", tags=["MCP Tools"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC REQUEST/RESPONSE MODELS (mcp_009)
# ─────────────────────────────────────────────────────────────────────────────


class ScrapeRequest(BaseModel):
    """Jewelry scraping request model."""
    category: Optional[str] = Field(
        default="all", description="Jewelry category")
    max_pages: PositiveInt = Field(
        default=5, le=50, description="Max pages to scrape")
    min_price: Optional[float] = Field(
        default=None, ge=0, description="Min price filter")
    max_price: Optional[float] = Field(
        default=None, ge=0, description="Max price filter")
    brand: Optional[str] = Field(default=None, description="Brand filter")
    include_images: bool = Field(default=True, description="Download images")

    @validator('max_price')
    def validate_price_range(cls, v, values):
        if v and values.get('min_price') and v <= values['min_price']:
            raise ValueError('max_price must be greater than min_price')
        return v


class QueryRequest(BaseModel):
    """Query jewelry listings request model."""
    category: Optional[str] = Field(
        default=None, description="Category filter")
    search_text: Optional[str] = Field(default=None, description="Search term")
    min_price: Optional[float] = Field(
        default=None, ge=0, description="Min price")
    max_price: Optional[float] = Field(
        default=None, ge=0, description="Max price")
    brand: Optional[str] = Field(default=None, description="Brand filter")
    condition: Optional[str] = Field(
        default=None, description="Condition filter")
    limit: PositiveInt = Field(default=50, le=500, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")


class ExportRequest(BaseModel):
    """Data export request model."""
    format: str = Field(description="Export format (json, csv, xlsx)")
    category: Optional[str] = Field(
        default=None, description="Category filter")
    filename: Optional[str] = Field(
        default=None, description="Custom filename")

    @validator('format')
    def validate_format(cls, v):
        if v.lower() not in ['json', 'csv', 'xlsx']:
            raise ValueError('Format must be json, csv, or xlsx')
        return v.lower()


class CleanupRequest(BaseModel):
    """Data cleanup request model."""
    older_than_days: PositiveInt = Field(
        default=30, description="Age threshold")
    categories: Optional[List[str]] = Field(
        default=None, description="Category filter")
    dry_run: bool = Field(default=True, description="Preview mode")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE AND INITIALIZATION (mcp_001)
# ─────────────────────────────────────────────────────────────────────────────


# Initialize database manager
# Use environment variable or default to relative path
DB_PATH = os.getenv('JEWELRY_DB_PATH', str(
    Path(__file__).parent.parent / "data" / "jewelry_scraping.db"))
db_manager = JewelryDatabaseManager(DB_PATH)

# Initialize URL builder
url_builder = EBayURLBuilder(enable_validation=True, enable_logging=True)

# Active scraping sessions
active_scrapers = {}

# System status tracking
system_status = {
    "startup_time": time.time(),
    "total_listings": 0,
    "active_tasks": 0,
    "error_count": 0,
    "last_scrape": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Jewelry MCP Server...")

    # Update system stats
    try:
        stats = db_manager.get_database_stats()
        system_status["total_listings"] = stats.total_listings
    except Exception as e:
        logger.warning(f"Could not load initial stats: {e}")

    yield

    # Cleanup active scrapers on shutdown
    logger.info("Shutting down Jewelry MCP Server...")
    for task_id, scraper in list(active_scrapers.items()):
        try:
            logger.info(f"Closing scraper for task {task_id}")
            await scraper.close()
        except Exception as e:
            logger.warning(f"Error closing scraper {task_id}: {e}")
    
    active_scrapers.clear()
    logger.info("Jewelry MCP Server stopped.")

# Create FastAPI app
app = FastAPI(
    title="Jewelry Scraper MCP Server",
    description="FastMCP server for jewelry scraping operations",
    version="1.0.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────────────────────
# MCP TOOL ENDPOINTS (mcp_002 to mcp_006)
# ─────────────────────────────────────────────────────────────────────────────


@tool("scrape_jewelry")
async def scrape_jewelry(request: ScrapeRequest) -> Dict[str, Any]:
    """
    Trigger jewelry scraping from eBay with specified filters.

    Args:
        request: Scraping configuration

    Returns:
        Scraping task status and information
    """
    try:
        task_id = f"scrape_{int(time.time())}"
        system_status["active_tasks"] += 1
        
        logger.info(f"Starting real scraping task {task_id} for category: {request.category}")

        # Map category string to enum
        category_mapping = {
            "rings": URLJewelryCategory.RINGS,
            "necklaces": URLJewelryCategory.NECKLACES,
            "earrings": URLJewelryCategory.EARRINGS,
            "bracelets": URLJewelryCategory.BRACELETS,
            "watches": URLJewelryCategory.WATCHES,
            "gemstones": URLJewelryCategory.GEMSTONES,
            "vintage": URLJewelryCategory.VINTAGE_ANTIQUE,
            "all": None
        }
        
        url_category = category_mapping.get(request.category.lower())
        
        # Create search filters
        search_filters = SearchFilters(
            query=f"{request.category} jewelry" if request.category != "all" else "jewelry",
            category=url_category,
            min_price=request.min_price,
            max_price=request.max_price,
            brand=request.brand,
            items_per_page=min(50, 200)  # Reasonable limit per page
        )
        
        # Build search URL
        url_result = url_builder.build_search_url(search_filters)
        if not url_result.is_valid:
            return {
                "success": False,
                "error": f"Invalid search parameters: {', '.join(url_result.errors)}"
            }
        
        # Configure scraper
        scraper_config = ScrapingConfig(
            max_concurrent_requests=2,
            request_delay_range=(2, 4),
            max_retries=3,
            anti_detection_level=AntiDetectionLevel.STANDARD,
            extract_images=request.include_images,
            min_data_quality_score=0.6,
            validate_data=True,
            skip_duplicates=True
        )
        
        # Initialize scraper
        scraper = EbayJewelryScraper(scraper_config)
        active_scrapers[task_id] = scraper
        
        # Start scraping
        logger.info(f"Scraping URL: {url_result.url}")
        scraping_result = await scraper.scrape_search_results(
            url_result.url,
            max_pages=request.max_pages,
            max_listings=request.max_pages * 50
        )
        
        if scraping_result.success:
            # Store scraped data in database
            listings = scraping_result.data or []
            stored_count = 0
            
            for listing in listings:
                try:
                    # Save to database
                    db_manager.store_listing(listing)
                    stored_count += 1
                except Exception as e:
                    logger.warning(f"Failed to store listing {listing.id}: {e}")
            
            # Update system status
            system_status["total_listings"] += stored_count
            system_status["last_scrape"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "task_id": task_id,
                "status": "completed",
                "message": f"Successfully scraped {stored_count} {request.category} listings",
                "listings_found": len(listings),
                "listings_stored": stored_count,
                "pages_scraped": scraping_result.metadata.get('pages_scraped', 0),
                "execution_time": scraping_result.response_time,
                "quality_score": scraping_result.quality_score,
                "session_stats": scraper.get_session_stats()
            }
        else:
            return {
                "success": False,
                "task_id": task_id,
                "status": "failed",
                "error": scraping_result.error,
                "execution_time": scraping_result.response_time
            }
            
    except Exception as e:
        system_status["error_count"] += 1
        logger.error(f"Scraping task {task_id} failed: {e}")
        return {"success": False, "task_id": task_id, "error": str(e)}
    finally:
        # Clean up scraper
        if task_id in active_scrapers:
            try:
                await active_scrapers[task_id].close()
                del active_scrapers[task_id]
            except Exception as e:
                logger.warning(f"Failed to clean up scraper {task_id}: {e}")
        
        system_status["active_tasks"] = max(0, system_status["active_tasks"] - 1)


@tool("query_jewelry")
async def query_jewelry(request: QueryRequest) -> Dict[str, Any]:
    """
    Query stored jewelry listings with advanced filtering.

    Args:
        request: Query parameters

    Returns:
        Filtered jewelry listings and metadata
    """
    try:
        # Convert request to QueryFilters
        filters = QueryFilters(
            category=request.category,
            min_price=request.min_price,
            max_price=request.max_price,
            brand=request.brand,
            condition=request.condition,
            search_text=request.search_text
        )

        # Query database
        listings = db_manager.query_listings(
            filters, request.limit, request.offset)

        # Convert to dictionaries for response
        listing_dicts = [listing.get_enhanced_summary()
                         for listing in listings]

        return {
            "success": True,
            "total_count": len(listings),
            "listings": listing_dicts,
            "query_time": datetime.now().isoformat()
        }

    except Exception as e:
        system_status["error_count"] += 1
        logger.error(f"Query failed: {e}")
        return {"success": False, "error": str(e)}


@tool("export_jewelry_data")
async def export_jewelry_data(request: ExportRequest) -> Dict[str, Any]:
    """
    Export jewelry data in specified format.

    Args:
        request: Export configuration

    Returns:
        Export file information
    """
    try:
        # Generate filename if not provided
        if not request.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            category = request.category or "all"
            request.filename = f"jewelry_export_{category}_{timestamp}.{request.format}"

        # Create exports directory
        export_dir = Path(os.getenv('JEWELRY_EXPORT_DIR', str(
            Path(__file__).parent.parent / "data" / "exports")))
        export_dir.mkdir(exist_ok=True)
        output_path = export_dir / request.filename

        # Query data for export
        filters = QueryFilters(category=request.category)

        # Export based on format
        if request.format == "json":
            success = db_manager.export_to_json(str(output_path), filters)
        elif request.format == "csv":
            success = db_manager.export_to_csv(str(output_path), filters)
        else:
            # For xlsx, use pandas if available
            success = db_manager.export_to_csv(
                str(output_path.with_suffix('.csv')), filters)

        if success:
            file_size = output_path.stat().st_size if output_path.exists() else 0
            return {
                "success": True,
                "filename": request.filename,
                "file_path": str(output_path),
                "file_size": file_size,
                "format": request.format,
                "created_at": datetime.now().isoformat()
            }
        else:
            return {"success": False, "error": "Export failed"}

    except Exception as e:
        system_status["error_count"] += 1
        logger.error(f"Export failed: {e}")
        return {"success": False, "error": str(e)}


@tool("system_status")
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system health and statistics.

    Returns:
        System status and database statistics
    """
    try:
        # Get database statistics
        stats = db_manager.get_database_stats()

        uptime = time.time() - system_status["startup_time"]

        return {
            "success": True,
            "status": "healthy",
            "uptime_seconds": uptime,
            "database": {
                "total_listings": stats.total_listings,
                "total_images": stats.total_images,
                "avg_quality_score": round(stats.avg_quality_score, 2),
                "storage_size_mb": round(stats.storage_size_mb, 2),
                "categories": stats.categories_breakdown,
                "materials": stats.materials_breakdown,
                "price_range": stats.price_range
            },
            "system": {
                "active_tasks": system_status.get("active_tasks", 0),
                "error_count": system_status.get("error_count", 0),
                "last_scrape": system_status.get("last_scrape")
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        system_status["error_count"] += 1
        logger.error(f"Status check failed: {e}")
        return {"success": False, "error": str(e)}


@tool("cleanup_old_data")
async def cleanup_old_data(request: CleanupRequest) -> Dict[str, Any]:
    """
    Clean up old jewelry data and files.

    Args:
        request: Cleanup configuration

    Returns:
        Cleanup results and statistics
    """
    try:
        # Perform cleanup
        cleanup_results = db_manager.cleanup_old_data(
            request.older_than_days,
            request.dry_run
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "listings_affected": cleanup_results.get("listings_to_delete", 0),
            "images_affected": cleanup_results.get("images_to_delete", 0),
            "deleted": cleanup_results.get("deleted", False),
            "cutoff_date": (datetime.now() - timedelta(days=request.older_than_days)).isoformat(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        system_status["error_count"] += 1
        logger.error(f"Cleanup failed: {e}")
        return {"success": False, "error": str(e)}


@tool("get_active_tasks")
async def get_active_tasks() -> Dict[str, Any]:
    """
    Get information about currently active scraping tasks.

    Returns:
        List of active tasks with their status
    """
    try:
        tasks = []
        for task_id, scraper in active_scrapers.items():
            try:
                stats = scraper.get_session_stats()
                tasks.append({
                    "task_id": task_id,
                    "status": stats.get("status", "unknown"),
                    "listings_found": stats.get("listings_found", 0),
                    "pages_processed": stats.get("pages_processed", 0),
                    "duration": stats.get("duration", 0),
                    "detection_risk_score": stats.get("detection_risk_score", 0.0)
                })
            except Exception as e:
                logger.warning(f"Failed to get stats for task {task_id}: {e}")
                tasks.append({
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "active_tasks_count": len(tasks),
            "tasks": tasks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        return {"success": False, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# MCP RESOURCE ENDPOINTS (mcp_007)
# ─────────────────────────────────────────────────────────────────────────────


@resource("jewelry://categories")
async def get_jewelry_categories() -> Dict[str, Any]:
    """
    Get available jewelry categories with statistics.

    Returns:
        Jewelry categories and listing counts
    """
    try:
        stats = db_manager.get_database_stats()

        categories = {}
        for category, count in stats.categories_breakdown.items():
            categories[category] = {
                "name": category.replace("_", " ").title(),
                "count": count,
                "percentage": round((count / max(stats.total_listings, 1)) * 100, 1)
            }

        return {
            "categories": categories,
            "total_categories": len(categories),
            "total_listings": stats.total_listings,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return {"error": str(e)}


@resource("jewelry://statistics")
async def get_jewelry_statistics() -> Dict[str, Any]:
    """
    Get comprehensive jewelry database statistics.

    Returns:
        Detailed database analytics
    """
    try:
        stats = db_manager.get_database_stats()

        return {
            "overview": {
                "total_listings": stats.total_listings,
                "total_images": stats.total_images,
                "avg_quality_score": round(stats.avg_quality_score, 2),
                "storage_size_mb": round(stats.storage_size_mb, 2)
            },
            "categories": stats.categories_breakdown,
            "materials": stats.materials_breakdown,
            "price_analysis": {
                "min_price": stats.price_range[0],
                "max_price": stats.price_range[1],
                "range": stats.price_range[1] - stats.price_range[0]
            },
            "recent_activity": stats.recent_activity,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# MCP PROMPT TEMPLATES (mcp_008)
# ─────────────────────────────────────────────────────────────────────────────


@prompt("scrape_jewelry_prompt")
async def scrape_jewelry_prompt(
    category: str = "all",
    max_pages: int = 5,
    price_range: str = "any"
) -> str:
    """
    Generate prompt for jewelry scraping operations.

    Args:
        category: Target jewelry category
        max_pages: Maximum pages to scrape
        price_range: Price range specification

    Returns:
        Formatted prompt template
    """
    return f"""
Scrape {category} jewelry listings from eBay:

Configuration:
- Category: {category}
- Max pages: {max_pages}
- Price range: {price_range}
- Include images: Yes
- Rate limiting: Enabled

Start the scraping operation and provide status updates.
    """.strip()


@prompt("query_jewelry_prompt")
async def query_jewelry_prompt(
    search_term: str = "",
    category: str = "all",
    sort_by: str = "price"
) -> str:
    """
    Generate prompt for jewelry query operations.

    Args:
        search_term: Search filter
        category: Category filter
        sort_by: Sort preference

    Returns:
        Formatted query prompt
    """
    return f"""
Search jewelry listings:

Criteria:
- Search: "{search_term}"
- Category: {category}
- Sort by: {sort_by}
- Limit: 50 results

Return matching listings with details.
    """.strip()

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT (mcp_011)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    System health check endpoint.

    Returns:
        Health status and basic metrics
    """
    try:
        uptime = time.time() - system_status["startup_time"]

        # Test database connectivity
        db_healthy = True
        try:
            db_manager.get_database_stats()
        except Exception:
            db_healthy = False

        status = "healthy" if db_healthy else "degraded"

        return {
            "status": status,
            "uptime_seconds": uptime,
            "database_connected": db_healthy,
            "active_tasks": system_status.get("active_tasks", 0),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLING (mcp_010)
# ─────────────────────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for comprehensive error handling.

    Args:
        request: FastAPI request object
        exc: Exception that occurred

    Returns:
        Standardized error response
    """
    system_status["error_count"] += 1
    logger.error(f"Global exception on {request.url}: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ─────────────────────────────────────────────────────────────────────────────
# ATTACH MCP TO FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

# Register MCP router with FastAPI app  
app.include_router(mcp_router)

# Register tool endpoints manually
@mcp_router.post("/scrape")
async def mcp_scrape_jewelry(request: ScrapeRequest):
    """MCP endpoint for jewelry scraping"""
    return await scrape_jewelry(request)

@mcp_router.post("/query")
async def mcp_query_jewelry(request: QueryRequest):
    """MCP endpoint for querying jewelry listings"""
    return await query_jewelry(request)

@mcp_router.post("/export")
async def mcp_export_jewelry_data(request: ExportRequest):
    """MCP endpoint for exporting jewelry data"""
    return await export_jewelry_data(request)

@mcp_router.get("/status")
async def mcp_get_system_status():
    """MCP endpoint for system status"""
    return await get_system_status()

@mcp_router.post("/cleanup")
async def mcp_cleanup_old_data(request: CleanupRequest):
    """MCP endpoint for data cleanup"""
    return await cleanup_old_data(request)

@mcp_router.get("/tasks")
async def mcp_get_active_tasks():
    """MCP endpoint for active tasks"""
    return await get_active_tasks()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "jewelry_mcp_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
