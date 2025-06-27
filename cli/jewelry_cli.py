#!/usr/bin/env python3
"""
Jewelry Scraping CLI - Command-line interface for eBay jewelry scraping system
Built on top of Crawl4AI with specialized jewelry extraction capabilities
"""

import click
import os
import sys
import json
import time
import asyncio
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.text import Text
import requests
import yaml

# Initialize rich console
console = Console()

# Configuration
DEFAULT_CONFIG = {
    "database": {
        "path": "jewelry_scraping.db",
        "backup_path": "backups/"
    },
    "scraping": {
        "rate_limit": 2.0,
        "max_retries": 3,
        "timeout": 30,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "categories": ["rings", "necklaces", "earrings", "bracelets", "watches", "pendants"]
    },
    "images": {
        "download_path": "images/",
        "max_size": "1920x1080",
        "quality": 85,
        "formats": ["jpg", "png", "webp"]
    },
    "mcp_server": {
        "host": "localhost",
        "port": 8000,
        "api_key": None
    },
    "dashboard": {
        "port": 8501,
        "host": "localhost",
        "update_interval": 5
    }
}


@dataclass
class JewelryListing:
    """Data model for jewelry listings"""
    id: str
    title: str
    price: float
    original_price: Optional[float]
    category: str
    subcategory: Optional[str]
    brand: Optional[str]
    material: Optional[str]
    condition: str
    seller: str
    seller_rating: Optional[float]
    listing_url: str
    image_urls: List[str]
    description: str
    specifications: Dict[str, Any]
    shipping_info: Dict[str, Any]
    return_policy: str
    scraped_at: datetime
    updated_at: datetime


class ConfigManager:
    """Manage configuration for the jewelry scraping system"""

    def __init__(self, config_path: str = "jewelry_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or DEFAULT_CONFIG
        else:
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG

    def save_config(self, config: Dict):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        self.config = config

    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class DatabaseManager:
    """Manage SQLite database operations"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    price REAL,
                    original_price REAL,
                    category TEXT,
                    subcategory TEXT,
                    brand TEXT,
                    material TEXT,
                    condition TEXT,
                    seller TEXT,
                    seller_rating REAL,
                    listing_url TEXT UNIQUE,
                    image_urls TEXT,
                    description TEXT,
                    specifications TEXT,
                    shipping_info TEXT,
                    return_policy TEXT,
                    scraped_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS scraping_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    parameters TEXT,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    results_count INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_category ON listings(category)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_price ON listings(price)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scraped_at ON listings(scraped_at)")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total listings
            cursor.execute("SELECT COUNT(*) FROM listings")
            total_listings = cursor.fetchone()[0]

            # Listings by category
            cursor.execute(
                "SELECT category, COUNT(*) FROM listings GROUP BY category")
            categories = dict(cursor.fetchall())

            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM listings 
                WHERE scraped_at > datetime('now', '-24 hours')
            """)
            recent_listings = cursor.fetchone()[0]

            return {
                "total_listings": total_listings,
                "categories": categories,
                "recent_listings": recent_listings,
                "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0
            }

    def search_listings(self, query: str = None, category: str = None,
                        min_price: float = None, max_price: float = None,
                        limit: int = 100) -> List[Dict]:
        """Search listings with filters"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            sql = "SELECT * FROM listings WHERE 1=1"
            params = []

            if query:
                sql += " AND (title LIKE ? OR description LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])

            if category:
                sql += " AND category = ?"
                params.append(category)

            if min_price is not None:
                sql += " AND price >= ?"
                params.append(min_price)

            if max_price is not None:
                sql += " AND price <= ?"
                params.append(max_price)

            sql += " ORDER BY scraped_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]


class MCPClient:
    """Client for MCP server communication"""

    def __init__(self, host: str, port: int, api_key: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def health_check(self) -> Dict[str, Any]:
        """Check MCP server health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_scraping_job(self, job_type: str, parameters: Dict) -> Dict[str, Any]:
        """Start a scraping job"""
        try:
            response = self.session.post(
                f"{self.base_url}/jobs/scrape",
                json={"job_type": job_type, "parameters": parameters},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        try:
            response = self.session.get(
                f"{self.base_url}/jobs/{job_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Global instances
config_manager = ConfigManager()
db_manager = DatabaseManager(config_manager.get(
    "database.path", "jewelry_scraping.db"))
mcp_client = MCPClient(
    config_manager.get("mcp_server.host", "localhost"),
    config_manager.get("mcp_server.port", 8000),
    config_manager.get("mcp_server.api_key")
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Jewelry Scraping CLI - Advanced eBay jewelry scraping system

    Built on top of Crawl4AI with specialized jewelry extraction capabilities.
    Features include automated scraping, image processing, data analytics,
    and real-time monitoring dashboard.

    Commands:
        scrape    - Start jewelry scraping operations
        list      - List and filter scraped jewelry data
        export    - Export jewelry data to various formats
        status    - Show system status and health
        cleanup   - Clean up old data and files
        serve     - Start MCP server
        deploy    - Deploy the entire system
        dashboard - Launch monitoring dashboard
        test      - Run system tests
        setup     - Initial system setup
    """
    pass


@cli.command()
@click.option('--category', '-c', type=click.Choice(['rings', 'necklaces', 'earrings', 'bracelets', 'watches', 'pendants']),
              help='Jewelry category to scrape')
@click.option('--search-term', '-s', help='Search term for jewelry')
@click.option('--max-pages', '-p', default=5, help='Maximum pages to scrape (default: 5)')
@click.option('--min-price', type=float, help='Minimum price filter')
@click.option('--max-price', type=float, help='Maximum price filter')
@click.option('--condition', type=click.Choice(['new', 'used', 'refurbished']), help='Item condition filter')
@click.option('--async-mode', '-a', is_flag=True, help='Run scraping asynchronously')
@click.option('--download-images', '-i', is_flag=True, help='Download product images')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def scrape(category, search_term, max_pages, min_price, max_price, condition, async_mode, download_images, verbose):
    """Start jewelry scraping operations"""

    console.print(Panel(
        "[bold cyan]Starting Jewelry Scraping Operation[/bold cyan]\n"
        f"Category: {category or 'All'}\n"
        f"Search Term: {search_term or 'None'}\n"
        f"Max Pages: {max_pages}\n"
        f"Mode: {'Async' if async_mode else 'Sync'}\n"
        f"Download Images: {'Yes' if download_images else 'No'}",
        title="Scraping Configuration",
        border_style="cyan"
    ))

    # Prepare job parameters
    job_params = {
        "category": category,
        "search_term": search_term,
        "max_pages": max_pages,
        "min_price": min_price,
        "max_price": max_price,
        "condition": condition,
        "download_images": download_images
    }

    # Start scraping job
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        task = progress.add_task("Initializing scraping job...", total=100)

        # Start job via MCP server
        job_result = mcp_client.start_scraping_job(
            "jewelry_scrape", job_params)
        progress.update(task, advance=20,
                        description="Job started, scraping pages...")

        if job_result.get("status") == "error":
            console.print(
                f"[red]Error starting job: {job_result.get('message')}[/red]")
            return

        job_id = job_result.get("job_id")

        # Monitor job progress
        while True:
            status = mcp_client.get_job_status(job_id)

            if status.get("status") == "completed":
                progress.update(task, completed=100,
                                description="Scraping completed!")
                console.print(
                    f"[green]✓ Scraping completed successfully![/green]")
                console.print(
                    f"[cyan]Results: {status.get('results_count', 0)} listings scraped[/cyan]")
                break
            elif status.get("status") == "error":
                console.print(
                    f"[red]✗ Scraping failed: {status.get('error_message')}[/red]")
                break
            elif status.get("status") == "running":
                progress.update(task, advance=10,
                                description="Scraping in progress...")
                time.sleep(2)
            else:
                time.sleep(1)


@cli.command()
@click.option('--category', '-c', help='Filter by category')
@click.option('--query', '-q', help='Search query')
@click.option('--min-price', type=float, help='Minimum price')
@click.option('--max-price', type=float, help='Maximum price')
@click.option('--limit', '-l', default=50, help='Maximum results to show (default: 50)')
@click.option('--sort', type=click.Choice(['price', 'date', 'title']), default='date', help='Sort by field')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'csv']), default='table', help='Output format')
def list(category, query, min_price, max_price, limit, sort, format):
    """List and filter scraped jewelry data"""

    # Search listings
    listings = db_manager.search_listings(
        query=query,
        category=category,
        min_price=min_price,
        max_price=max_price,
        limit=limit
    )

    if not listings:
        console.print(
            "[yellow]No listings found matching your criteria.[/yellow]")
        return

    if format == 'json':
        click.echo(json.dumps(listings, indent=2, default=str))
    elif format == 'csv':
        # Simple CSV output
        if listings:
            headers = listings[0].keys()
            click.echo(','.join(headers))
            for listing in listings:
                click.echo(','.join(str(listing.get(h, '')) for h in headers))
    else:
        # Rich table format
        table = Table(title=f"Jewelry Listings ({len(listings)} found)")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green", max_width=40)
        table.add_column("Category", style="blue")
        table.add_column("Price", style="yellow", justify="right")
        table.add_column("Condition", style="magenta")
        table.add_column("Scraped", style="dim")

        for listing in listings:
            table.add_row(
                listing.get('id', '')[:8],
                listing.get('title', '')[
                    :37] + '...' if len(listing.get('title', '')) > 40 else listing.get('title', ''),
                listing.get('category', ''),
                f"${listing.get('price', 0):.2f}",
                listing.get('condition', ''),
                listing.get('scraped_at', '')[
                    :10] if listing.get('scraped_at') else ''
            )

        console.print(table)


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'xlsx']), default='json', help='Export format')
@click.option('--output', '-o', help='Output file path')
@click.option('--category', '-c', help='Filter by category')
@click.option('--query', '-q', help='Search query')
@click.option('--min-price', type=float, help='Minimum price')
@click.option('--max-price', type=float, help='Maximum price')
def export(format, output, category, query, min_price, max_price):
    """Export jewelry data to various formats"""

    # Get listings
    listings = db_manager.search_listings(
        query=query,
        category=category,
        min_price=min_price,
        max_price=max_price,
        limit=10000  # Large limit for export
    )

    if not listings:
        console.print("[yellow]No data to export.[/yellow]")
        return

    # Generate output filename if not provided
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"jewelry_export_{timestamp}.{format}"

    # Export based on format
    if format == 'json':
        with open(output, 'w') as f:
            json.dump(listings, f, indent=2, default=str)
    elif format == 'csv':
        import csv
        if listings:
            with open(output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=listings[0].keys())
                writer.writeheader()
                writer.writerows(listings)
    elif format == 'xlsx':
        try:
            import pandas as pd
            df = pd.DataFrame(listings)
            df.to_excel(output, index=False)
        except ImportError:
            console.print(
                "[red]pandas and openpyxl required for Excel export. Install with: pip install pandas openpyxl[/red]")
            return

    console.print(
        f"[green]✓ Exported {len(listings)} listings to {output}[/green]")


@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed status information')
def status(detailed):
    """Show system status and health"""

    console.print(Panel("[bold cyan]System Status Check[/bold cyan]",
                  title="Status", border_style="cyan"))

    with console.status("[bold green]Checking system health..."):
        # Database status
        db_stats = db_manager.get_stats()

        # MCP server status
        mcp_health = mcp_client.health_check()

        # File system status
        config_path = Path(config_manager.config_path)
        images_path = Path(config_manager.get(
            "images.download_path", "images/"))

    # Create status table
    table = Table(title="System Health")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    # Database status
    db_status = "✓ Online" if db_stats["total_listings"] >= 0 else "✗ Error"
    table.add_row(
        "Database",
        db_status,
        f"{db_stats['total_listings']} listings, {db_stats['recent_listings']} recent"
    )

    # MCP Server status
    mcp_status = "✓ Online" if mcp_health.get(
        "status") == "ok" else "✗ Offline"
    table.add_row(
        "MCP Server",
        mcp_status,
        mcp_health.get("message", "Ready")
    )

    # File system
    config_status = "✓ Available" if config_path.exists() else "✗ Missing"
    images_status = "✓ Available" if images_path.exists() else "✗ Missing"
    table.add_row("Configuration", config_status, str(config_path))
    table.add_row("Images Directory", images_status, str(images_path))

    console.print(table)

    if detailed:
        # Show detailed statistics
        console.print("\n[bold]Detailed Statistics:[/bold]")

        # Categories breakdown
        if db_stats["categories"]:
            cat_table = Table(title="Listings by Category")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green", justify="right")

            for category, count in db_stats["categories"].items():
                cat_table.add_row(category, str(count))

            console.print(cat_table)


@cli.command()
@click.option('--older-than', '-o', default=30, help='Delete data older than N days (default: 30)')
@click.option('--dry-run', '-d', is_flag=True, help='Show what would be deleted without actually deleting')
@click.option('--category', '-c', help='Only cleanup specific category')
@click.option('--images', '-i', is_flag=True, help='Also cleanup unused images')
@click.confirmation_option(prompt='Are you sure you want to cleanup old data?')
def cleanup(older_than, dry_run, category, images):
    """Clean up old data and files"""

    console.print(Panel(
        f"[bold yellow]Cleanup Operation[/bold yellow]\n"
        f"Older than: {older_than} days\n"
        f"Category: {category or 'All'}\n"
        f"Include images: {'Yes' if images else 'No'}\n"
        f"Dry run: {'Yes' if dry_run else 'No'}",
        title="Cleanup Configuration",
        border_style="yellow"
    ))

    if dry_run:
        console.print(
            "[yellow]DRY RUN - No data will be actually deleted[/yellow]")

    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()

        # Build cleanup query
        sql = "SELECT id, title, category FROM listings WHERE scraped_at < datetime('now', '-{} days')".format(
            older_than)
        params = []

        if category:
            sql += " AND category = ?"
            params.append(category)

        cursor.execute(sql, params)
        to_delete = cursor.fetchall()

        if not to_delete:
            console.print("[green]No old data found to cleanup.[/green]")
            return

        console.print(
            f"[yellow]Found {len(to_delete)} listings to cleanup:[/yellow]")

        # Show what will be deleted
        for listing_id, title, cat in to_delete[:10]:  # Show first 10
            console.print(f"  - {listing_id[:8]}: {title[:40]}... ({cat})")

        if len(to_delete) > 10:
            console.print(f"  ... and {len(to_delete) - 10} more")

        if not dry_run:
            # Delete old listings
            delete_sql = "DELETE FROM listings WHERE scraped_at < datetime('now', '-{} days')".format(
                older_than)
            if category:
                delete_sql += " AND category = ?"

            cursor.execute(delete_sql, params)
            conn.commit()

            console.print(
                f"[green]✓ Deleted {cursor.rowcount} old listings[/green]")

            # Cleanup images if requested
            if images:
                images_path = Path(config_manager.get(
                    "images.download_path", "images/"))
                if images_path.exists():
                    # This would need more sophisticated logic to identify unused images
                    console.print(
                        "[yellow]Image cleanup not implemented yet[/yellow]")


@cli.command()
@click.option('--host', '-h', default='localhost', help='Server host (default: localhost)')
@click.option('--port', '-p', default=8000, help='Server port (default: 8000)')
@click.option('--reload', '-r', is_flag=True, help='Enable auto-reload for development')
@click.option('--workers', '-w', default=1, help='Number of worker processes (default: 1)')
def serve(host, port, reload, workers):
    """Start MCP server"""

    console.print(Panel(
        f"[bold green]Starting MCP Server[/bold green]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}\n"
        f"Reload: {'Yes' if reload else 'No'}",
        title="MCP Server",
        border_style="green"
    ))

    # Check if server is already running
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        if response.status_code == 200:
            console.print("[yellow]Server is already running![/yellow]")
            return
    except:
        pass  # Server not running, continue with startup

    # Start server using the existing server.py
    server_path = Path("deploy/docker/server.py")
    if not server_path.exists():
        console.print(f"[red]Server file not found at {server_path}[/red]")
        return

    try:
        console.print("[cyan]Starting server...[/cyan]")

        # Run the server
        cmd = [
            sys.executable, str(server_path),
            "--host", host,
            "--port", str(port)
        ]

        if reload:
            cmd.append("--reload")

        subprocess.run(cmd)

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")


@cli.command()
@click.option('--environment', '-e', type=click.Choice(['development', 'production']), default='development',
              help='Deployment environment (default: development)')
@click.option('--build', '-b', is_flag=True, help='Build Docker images before deployment')
@click.option('--logs', '-l', is_flag=True, help='Show logs after deployment')
def deploy(environment, build, logs):
    """Deploy the entire system"""

    console.print(Panel(
        f"[bold blue]System Deployment[/bold blue]\n"
        f"Environment: {environment}\n"
        f"Build images: {'Yes' if build else 'No'}\n"
        f"Show logs: {'Yes' if logs else 'No'}",
        title="Deployment",
        border_style="blue"
    ))

    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"],
                       capture_output=True, check=True)
        subprocess.run(["docker-compose", "--version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print(
            "[red]Docker or docker-compose not found. Please install Docker first.[/red]")
        return

    # Check if docker-compose.yml exists
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        console.print(
            f"[red]docker-compose.yml not found at {compose_file}[/red]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            if build:
                task = progress.add_task(
                    "Building Docker images...", total=None)
                result = subprocess.run(
                    ["docker-compose", "build"], capture_output=True, text=True)
                if result.returncode != 0:
                    console.print(f"[red]Build failed: {result.stderr}[/red]")
                    return
                progress.update(task, description="Build completed")

            task = progress.add_task("Starting services...", total=None)
            result = subprocess.run(
                ["docker-compose", "up", "-d"], capture_output=True, text=True)
            if result.returncode != 0:
                console.print(f"[red]Deployment failed: {result.stderr}[/red]")
                return

            progress.update(task, description="Services started")

        console.print("[green]✓ Deployment completed successfully![/green]")

        # Show service status
        result = subprocess.run(
            ["docker-compose", "ps"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("\n[bold]Service Status:[/bold]")
            console.print(result.stdout)

        if logs:
            console.print("\n[bold]Service Logs:[/bold]")
            subprocess.run(["docker-compose", "logs", "-f"])

    except Exception as e:
        console.print(f"[red]Deployment error: {e}[/red]")


@cli.command()
@click.option('--port', '-p', default=8501, help='Dashboard port (default: 8501)')
@click.option('--host', '-h', default='localhost', help='Dashboard host (default: localhost)')
@click.option('--dev', '-d', is_flag=True, help='Run in development mode')
def dashboard(port, host, dev):
    """Launch monitoring dashboard"""

    console.print(Panel(
        f"[bold purple]Starting Monitoring Dashboard[/bold purple]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Mode: {'Development' if dev else 'Production'}\n"
        f"URL: http://{host}:{port}",
        title="Dashboard",
        border_style="purple"
    ))

    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        console.print(
            "[red]Streamlit not found. Install with: pip install streamlit[/red]")
        return

    # Create dashboard script if it doesn't exist
    dashboard_script = Path("jewelry_dashboard.py")
    if not dashboard_script.exists():
        console.print("[yellow]Creating dashboard script...[/yellow]")
        # We'll create this in the next step

    try:
        console.print("[cyan]Starting dashboard...[/cyan]")
        console.print(
            f"[dim]Dashboard will be available at http://{host}:{port}[/dim]")

        cmd = [
            "streamlit", "run", str(dashboard_script),
            "--server.port", str(port),
            "--server.address", host
        ]

        if not dev:
            cmd.extend([
                "--server.headless", "true",
                "--browser.serverAddress", host
            ])

        subprocess.run(cmd)

    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting dashboard: {e}[/red]")


@cli.command()
@click.option('--component', '-c', type=click.Choice(['scraper', 'database', 'mcp', 'images', 'all']),
              default='all', help='Component to test (default: all)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose test output')
@click.option('--coverage', is_flag=True, help='Generate coverage report')
def test(component, verbose, coverage):
    """Run system tests"""

    console.print(Panel(
        f"[bold green]Running System Tests[/bold green]\n"
        f"Component: {component}\n"
        f"Verbose: {'Yes' if verbose else 'No'}\n"
        f"Coverage: {'Yes' if coverage else 'No'}",
        title="Testing",
        border_style="green"
    ))

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        console.print(
            "[red]pytest not found. Install with: pip install pytest[/red]")
        return

    # Test configuration
    test_files = []
    if component == 'all':
        test_files = ["tests/"]
    else:
        test_files = [f"tests/test_{component}.py"]

    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    cmd.extend(test_files)

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])

    try:
        console.print("[cyan]Running tests...[/cyan]")
        result = subprocess.run(cmd)

        if result.returncode == 0:
            console.print("[green]✓ All tests passed![/green]")
        else:
            console.print("[red]✗ Some tests failed[/red]")
            sys.exit(result.returncode)

    except Exception as e:
        console.print(f"[red]Error running tests: {e}[/red]")


@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force setup even if already configured')
@click.option('--dev', '-d', is_flag=True, help='Setup for development environment')
def setup(force, dev):
    """Initial system setup"""

    console.print(Panel(
        f"[bold yellow]System Setup[/bold yellow]\n"
        f"Force: {'Yes' if force else 'No'}\n"
        f"Development: {'Yes' if dev else 'No'}",
        title="Setup",
        border_style="yellow"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        # Check existing setup
        task = progress.add_task("Checking existing setup...", total=100)

        config_exists = Path(config_manager.config_path).exists()
        db_exists = Path(db_manager.db_path).exists()

        if config_exists and db_exists and not force:
            console.print(
                "[yellow]System already configured. Use --force to reconfigure.[/yellow]")
            return

        progress.update(task, advance=20,
                        description="Creating directories...")

        # Create necessary directories
        directories = [
            "data",
            "images",
            "backups",
            "logs",
            "tests"
        ]

        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)

        progress.update(task, advance=20,
                        description="Initializing database...")

        # Initialize database
        db_manager._init_database()

        progress.update(task, advance=20,
                        description="Creating configuration...")

        # Create configuration
        config = DEFAULT_CONFIG.copy()
        if dev:
            # Allow external access in dev
            config["dashboard"]["host"] = "0.0.0.0"
            config["mcp_server"]["host"] = "0.0.0.0"

        config_manager.save_config(config)

        progress.update(task, advance=20,
                        description="Installing dependencies...")

        # Create requirements.txt if it doesn't exist
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            requirements = [
                "click>=8.0.0",
                "rich>=10.0.0",
                "requests>=2.25.0",
                "pyyaml>=5.4.0",
                "streamlit>=1.0.0",
                "plotly>=5.0.0",
                "pandas>=1.3.0",
                "crawl4ai>=0.3.0",
                "fastapi>=0.68.0",
                "uvicorn>=0.15.0"
            ]

            with open(requirements_file, 'w') as f:
                f.write('\n'.join(requirements))

        progress.update(task, advance=20, description="Setup completed!")
        progress.update(task, completed=100)

    console.print("[green]✓ System setup completed successfully![/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print(
        "1. Start the MCP server: [cyan]python jewelry_cli.py serve[/cyan]")
    console.print(
        "2. Launch the dashboard: [cyan]python jewelry_cli.py dashboard[/cyan]")
    console.print(
        "3. Start scraping: [cyan]python jewelry_cli.py scrape --category rings[/cyan]")


def main():
    """Main entry point for the CLI"""
    cli()

if __name__ == '__main__':
    main()
