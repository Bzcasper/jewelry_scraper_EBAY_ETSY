#!/usr/bin/env python3
"""
Simple Jewelry Scraper API Server
=================================

FastAPI server to trigger jewelry scraping operations via HTTP endpoints.
Simplified version without complex imports.
"""

import asyncio
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Jewelry Scraper API",
    description="Simple API to trigger jewelry scraping operations",
    version="1.0.0"
)

# Request/Response models
class ScrapeRequest(BaseModel):
    categories: Optional[List[str]] = ["rings", "necklaces"]
    max_items: Optional[int] = 10
    
class ScrapeResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None
    
class StatusResponse(BaseModel):
    status: str
    database_listings: int
    last_scrape: Optional[str] = None

# Global job tracking
active_jobs = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Jewelry Scraper API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/scrape", "method": "POST", "description": "Trigger scraping"},
            {"path": "/status", "method": "GET", "description": "System status"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/scrape", response_model=ScrapeResponse)
async def trigger_scrape(request: ScrapeRequest):
    """Trigger jewelry scraping operation"""
    try:
        job_id = f"scrape_{int(datetime.now().timestamp())}"
        
        # Store job info
        active_jobs[job_id] = {
            "status": "started",
            "categories": request.categories,
            "started_at": datetime.now().isoformat()
        }
        
        # Run scraper in background
        scraper_path = Path(__file__).parent / "run_jewelry_scraper.py"
        
        # Start scraping process
        process = subprocess.Popen([
            sys.executable, str(scraper_path), "scrape"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Update job status
        active_jobs[job_id]["process"] = process
        active_jobs[job_id]["status"] = "running"
        
        return ScrapeResponse(
            status="success",
            message=f"Scraping job started for categories: {request.categories}",
            job_id=job_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    try:
        # Check database
        import sqlite3
        db_path = Path("jewelry_scraping.db")
        
        listings_count = 0
        last_scrape = None
        
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM listings")
                listings_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT MAX(scraped_at) FROM listings")
                last_scrape = cursor.fetchone()[0]
        
        return StatusResponse(
            status="operational",
            database_listings=listings_count,
            last_scrape=last_scrape
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    # Check if process is still running
    if "process" in job:
        process = job["process"]
        if process.poll() is None:
            job["status"] = "running"
        else:
            job["status"] = "completed" if process.returncode == 0 else "failed"
            job["completed_at"] = datetime.now().isoformat()
    
    return job

if __name__ == "__main__":
    print("🚀 Starting Jewelry Scraper API Server")
    print("📡 Available at: http://localhost:8000")
    print("📚 API Docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "simple_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )