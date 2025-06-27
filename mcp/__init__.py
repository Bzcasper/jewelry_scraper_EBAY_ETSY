"""
Jewelry Scraper MCP Module
===========================

FastMCP server implementation for natural language control of the 
eBay jewelry scraping system.

This module provides:
- MCP tool endpoints for scraping, querying, exporting
- Resource endpoints for accessing data and statistics  
- Prompt templates for common operations
- Comprehensive error handling and validation
- Health monitoring and status reporting

Usage:
    from jewelry_scraper.mcp import JewelryMCPServer
    
    # Start the MCP server
    server = JewelryMCPServer()
    await server.start()
"""

from .jewelry_mcp_server import app, mcp_router

__version__ = "1.0.0"
__author__ = "Jewelry Scraper Team"

__all__ = [
    "app",
    "mcp_router"
]