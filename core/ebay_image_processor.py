"""
High-Performance Image Processing Pipeline for eBay Jewelry Images

This module provides concurrent image processing capabilities including:
- Async image downloading with rate limiting
- Image optimization and enhancement
- Metadata generation and validation
- Automatic categorization and cleanup
"""

import asyncio
import aiohttp
import aiofiles
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

from PIL import Image, ImageEnhance, ImageOps, ExifTags
from PIL.ExifTags import TAGS
import requests


class JewelryCategory(Enum):
    """Jewelry categories for organization"""
    RINGS = "rings"
    NECKLACES = "necklaces"
    EARRINGS = "earrings"
    BRACELETS = "bracelets"
    WATCHES = "watches"
    OTHER = "other"


class ImageQuality(Enum):
    """Image quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class ImageMetadata:
    """Comprehensive image metadata container"""
    filename: str
    url: str
    file_size: int
    dimensions: Tuple[int, int]
    format: str
    mode: str
    file_hash: str
    download_timestamp: datetime
    processing_timestamp: Optional[datetime] = None
    category: Optional[JewelryCategory] = None
    quality_score: Optional[float] = None
    exif_data: Optional[Dict] = None
    is_optimized: bool = False
    compression_ratio: Optional[float] = None
    validation_status: str = "pending"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['download_timestamp'] = self.download_timestamp.isoformat()
        if self.processing_timestamp:
            data['processing_timestamp'] = self.processing_timestamp.isoformat()
        if self.category:
            data['category'] = self.category.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['download_timestamp'] = datetime.fromisoformat(data['download_timestamp'])
        if data.get('processing_timestamp'):
            data['processing_timestamp'] = datetime.fromisoformat(data['processing_timestamp'])
        if data.get('category'):
            data['category'] = JewelryCategory(data['category'])
        return cls(**data)


class ImageProcessor:
    """
    High-performance concurrent image processing pipeline for jewelry images
    """
    
    def __init__(self, 
                 base_directory: str = "./images",
                 max_concurrent_downloads: int = 10,
                 max_concurrent_processing: int = 5,
                 request_delay: float = 0.5,
                 timeout: int = 30,
                 max_retries: int = 3,
                 quality_settings: Optional[Dict[ImageQuality, Dict]] = None):
        """
        Initialize the image processor
        
        Args:
            base_directory: Base directory for image storage
            max_concurrent_downloads: Maximum concurrent downloads
            max_concurrent_processing: Maximum concurrent processing operations
            request_delay: Delay between requests (rate limiting)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed downloads
            quality_settings: Custom quality settings for optimization
        """
        self.base_directory = Path(base_directory)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_processing = max_concurrent_processing
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Semaphores for concurrency control
        self.download_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_processing)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        
        # Statistics tracking
        self.stats = {
            'downloaded': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_size': 0,
            'start_time': None
        }
        
        # Quality settings for different optimization levels
        self.quality_settings = quality_settings or {
            ImageQuality.LOW: {'max_size': (400, 400), 'quality': 60, 'optimize': True},
            ImageQuality.MEDIUM: {'max_size': (800, 800), 'quality': 75, 'optimize': True},
            ImageQuality.HIGH: {'max_size': (1200, 1200), 'quality': 85, 'optimize': True},
            ImageQuality.ULTRA: {'max_size': (2000, 2000), 'quality': 95, 'optimize': True}
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize directory structure
        self._initialize_directories()
        
        # Cache for processed images (to avoid reprocessing)
        self.processed_cache: Set[str] = set()
        self._load_processed_cache()
    
    def _initialize_directories(self):
        """Create directory structure for organized storage"""
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
        # Create category directories
        for category in JewelryCategory:
            category_dir = self.base_directory / category.value
            category_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for different quality levels
            for quality in ImageQuality:
                quality_dir = category_dir / quality.value
                quality_dir.mkdir(exist_ok=True)
        
        # Create metadata directory
        (self.base_directory / "metadata").mkdir(exist_ok=True)
        
        # Create cache directory
        (self.base_directory / "cache").mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized directory structure at {self.base_directory}")
    
    def _load_processed_cache(self):
        """Load cache of previously processed images"""
        cache_file = self.base_directory / "cache" / "processed.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.processed_cache = set(cache_data.get('processed', []))
                self.logger.info(f"Loaded {len(self.processed_cache)} processed images from cache")
            except Exception as e:
                self.logger.warning(f"Failed to load processed cache: {e}")
    
    def _save_processed_cache(self):
        """Save cache of processed images"""
        cache_file = self.base_directory / "cache" / "processed.json"
        try:
            cache_data = {
                'processed': list(self.processed_cache),
                'last_updated': datetime.now().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save processed cache: {e}")
    
    async def _rate_limit(self):
        """Implement rate limiting for requests"""
        async with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.request_delay:
                sleep_time = self.request_delay - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _extract_exif_data(self, image: Image.Image) -> Optional[Dict]:
        """Extract EXIF data from image"""
        try:
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
            return exif_data if exif_data else None
        except Exception as e:
            self.logger.debug(f"Failed to extract EXIF data: {e}")
            return None
    
    def _categorize_image(self, url: str, metadata: Optional[Dict] = None) -> JewelryCategory:
        """
        Automatically categorize jewelry image based on URL and metadata
        This is a basic implementation - can be enhanced with ML models
        """
        url_lower = url.lower()
        
        # Basic keyword matching for categorization
        if any(keyword in url_lower for keyword in ['ring', 'band', 'engagement', 'wedding']):
            return JewelryCategory.RINGS
        elif any(keyword in url_lower for keyword in ['necklace', 'chain', 'pendant', 'choker']):
            return JewelryCategory.NECKLACES
        elif any(keyword in url_lower for keyword in ['earring', 'stud', 'hoop', 'drop']):
            return JewelryCategory.EARRINGS
        elif any(keyword in url_lower for keyword in ['bracelet', 'bangle', 'cuff']):
            return JewelryCategory.BRACELETS
        elif any(keyword in url_lower for keyword in ['watch', 'timepiece', 'rolex', 'omega']):
            return JewelryCategory.WATCHES
        else:
            return JewelryCategory.OTHER
    
    def _calculate_quality_score(self, image: Image.Image) -> float:
        """Calculate quality score for an image (0.0 to 1.0)"""
        try:
            width, height = image.size
            
            # Base score on resolution
            resolution_score = min(1.0, (width * height) / (1000 * 1000))  # Normalize to 1MP
            
            # Consider aspect ratio (jewelry images work better with certain ratios)
            aspect_ratio = width / height
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.3  # Prefer square-ish images
            aspect_score = max(0.0, min(1.0, aspect_score))
            
            # Simple sharpness estimation (would need more sophisticated approach for production)
            # For now, just consider if image is too small
            size_penalty = 0.0
            if width < 200 or height < 200:
                size_penalty = 0.3
            
            quality_score = (resolution_score * 0.6 + aspect_score * 0.4) - size_penalty
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate quality score: {e}")
            return 0.5  # Default score
    
    def _validate_image(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate downloaded image file"""
        try:
            # Check if file exists and has content
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False, "File is empty or doesn't exist"
            
            # Try to open and verify the image
            with Image.open(file_path) as img:
                img.verify()  # Verify image integrity
                
                # Reopen for further checks (verify() closes the file)
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # Check minimum dimensions
                    if width < 50 or height < 50:
                        return False, "Image too small (minimum 50x50 pixels)"
                    
                    # Check maximum dimensions (prevent extremely large images)
                    if width > 5000 or height > 5000:
                        return False, "Image too large (maximum 5000x5000 pixels)"
                    
                    # Check supported formats
                    if img.format not in ['JPEG', 'PNG', 'WEBP', 'GIF']:
                        return False, f"Unsupported format: {img.format}"
            
            return True, None
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    async def download_image(self, 
                           url: str, 
                           filename: Optional[str] = None,
                           category: Optional[JewelryCategory] = None) -> Optional[ImageMetadata]:
        """
        Download a single image with rate limiting and validation
        
        Args:
            url: Image URL to download
            filename: Optional custom filename
            category: Optional jewelry category
            
        Returns:
            ImageMetadata object if successful, None if failed
        """
        async with self.download_semaphore:
            await self._rate_limit()
            
            # Generate filename if not provided
            if not filename:
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"image_{url_hash}"
            
            # Auto-categorize if not provided
            if not category:
                category = self._categorize_image(url)
            
            # Determine file path
            temp_path = self.base_directory / "cache" / f"{filename}_temp"
            
            # Check if already processed
            if url in self.processed_cache:
                self.stats['skipped'] += 1
                self.logger.debug(f"Skipping already processed image: {url}")
                return None
            
            for attempt in range(self.max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.read()
                                
                                # Save to temporary file
                                async with aiofiles.open(temp_path, 'wb') as f:
                                    await f.write(content)
                                
                                # Validate the downloaded image
                                is_valid, error_msg = self._validate_image(temp_path)
                                
                                if not is_valid:
                                    self.logger.warning(f"Invalid image from {url}: {error_msg}")
                                    temp_path.unlink(missing_ok=True)
                                    self.stats['failed'] += 1
                                    return None
                                
                                # Create metadata
                                metadata = ImageMetadata(
                                    filename=filename,
                                    url=url,
                                    file_size=len(content),
                                    dimensions=(0, 0),  # Will be updated after image processing
                                    format="",  # Will be updated after image processing
                                    mode="",  # Will be updated after image processing
                                    file_hash="",  # Will be calculated after final processing
                                    download_timestamp=datetime.now(),
                                    category=category,
                                    validation_status="valid"
                                )
                                
                                # Process the image
                                processed_metadata = await self._process_image(temp_path, metadata)
                                
                                if processed_metadata:
                                    self.processed_cache.add(url)
                                    self.stats['downloaded'] += 1
                                    self.stats['total_size'] += len(content)
                                    return processed_metadata
                                else:
                                    temp_path.unlink(missing_ok=True)
                                    self.stats['failed'] += 1
                                    return None
                            
                            else:
                                self.logger.warning(f"HTTP {response.status} for {url}")
                                
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout downloading {url} (attempt {attempt + 1})")
                except Exception as e:
                    self.logger.warning(f"Error downloading {url} (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Clean up temp file on failure
            temp_path.unlink(missing_ok=True)
            self.stats['failed'] += 1
            return None
    
    async def _process_image(self, temp_path: Path, metadata: ImageMetadata) -> Optional[ImageMetadata]:
        """Process downloaded image (resize, optimize, generate metadata)"""
        async with self.processing_semaphore:
            try:
                # Open image for processing
                with Image.open(temp_path) as img:
                    # Update metadata with image information
                    metadata.dimensions = img.size
                    metadata.format = img.format
                    metadata.mode = img.mode
                    metadata.exif_data = self._extract_exif_data(img)
                    metadata.quality_score = self._calculate_quality_score(img)
                    metadata.processing_timestamp = datetime.now()
                    
                    # Create optimized versions for different quality levels
                    for quality_level in ImageQuality:
                        await self._create_optimized_version(img, metadata, quality_level)
                
                # Calculate file hash of original
                metadata.file_hash = self._calculate_file_hash(temp_path)
                
                # Move original to appropriate category directory
                final_path = self.base_directory / metadata.category.value / f"{metadata.filename}_original.{metadata.format.lower()}"
                temp_path.rename(final_path)
                
                # Save metadata
                await self._save_metadata(metadata)
                
                self.stats['processed'] += 1
                self.logger.info(f"Successfully processed image: {metadata.filename}")
                
                return metadata
                
            except Exception as e:
                self.logger.error(f"Error processing image {temp_path}: {e}")
                temp_path.unlink(missing_ok=True)
                return None
    
    async def _create_optimized_version(self, img: Image.Image, metadata: ImageMetadata, quality_level: ImageQuality):
        """Create optimized version of image for specific quality level"""
        try:
            settings = self.quality_settings[quality_level]
            max_size = settings['max_size']
            quality = settings['quality']
            optimize = settings['optimize']
            
            # Create a copy for processing
            processed_img = img.copy()
            
            # Resize if necessary
            if processed_img.size[0] > max_size[0] or processed_img.size[1] > max_size[1]:
                processed_img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Enhance image quality
            if quality_level in [ImageQuality.HIGH, ImageQuality.ULTRA]:
                # Apply subtle enhancements for higher quality versions
                enhancer = ImageEnhance.Sharpness(processed_img)
                processed_img = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(1.05)
            
            # Auto-orient based on EXIF data
            processed_img = ImageOps.exif_transpose(processed_img)
            
            # Save optimized version
            output_path = (self.base_directory / metadata.category.value / quality_level.value / 
                          f"{metadata.filename}_{quality_level.value}.jpg")
            
            # Convert to RGB if necessary (for JPEG saving)
            if processed_img.mode in ('RGBA', 'LA', 'P'):
                processed_img = processed_img.convert('RGB')
            
            processed_img.save(
                output_path,
                'JPEG',
                quality=quality,
                optimize=optimize,
                progressive=True
            )
            
            # Calculate compression ratio for the highest quality version
            if quality_level == ImageQuality.HIGH:
                original_size = metadata.file_size
                compressed_size = output_path.stat().st_size
                metadata.compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                metadata.is_optimized = True
            
        except Exception as e:
            self.logger.error(f"Error creating {quality_level.value} version: {e}")
    
    async def _save_metadata(self, metadata: ImageMetadata):
        """Save image metadata to JSON file"""
        try:
            metadata_dir = self.base_directory / "metadata"
            metadata_file = metadata_dir / f"{metadata.filename}_metadata.json"
            
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata.to_dict(), indent=2))
                
        except Exception as e:
            self.logger.error(f"Error saving metadata for {metadata.filename}: {e}")
    
    async def process_urls(self, urls: List[str], 
                          batch_size: int = 50,
                          progress_callback: Optional[callable] = None) -> List[ImageMetadata]:
        """
        Process multiple image URLs concurrently
        
        Args:
            urls: List of image URLs to process
            batch_size: Process URLs in batches to manage memory
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of successfully processed ImageMetadata objects
        """
        self.stats['start_time'] = time.time()
        successful_metadata = []
        
        self.logger.info(f"Starting processing of {len(urls)} images")
        
        # Process URLs in batches
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            batch_tasks = [self.download_image(url) for url in batch_urls]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, ImageMetadata):
                        successful_metadata.append(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Batch processing error: {result}")
                
                # Call progress callback if provided
                if progress_callback:
                    progress = (i + len(batch_urls)) / len(urls)
                    progress_callback(progress, self.stats)
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                continue
        
        # Save processed cache
        self._save_processed_cache()
        
        elapsed_time = time.time() - self.stats['start_time']
        self.logger.info(f"Processing completed in {elapsed_time:.2f}s. "
                        f"Success: {len(successful_metadata)}, "
                        f"Failed: {self.stats['failed']}, "
                        f"Skipped: {self.stats['skipped']}")
        
        return successful_metadata
    
    async def cleanup_old_images(self, days_old: int = 30) -> int:
        """
        Clean up images older than specified days
        
        Args:
            days_old: Remove images older than this many days
            
        Returns:
            Number of images cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        try:
            # Clean up images in all category directories
            for category in JewelryCategory:
                category_dir = self.base_directory / category.value
                
                if category_dir.exists():
                    for file_path in category_dir.rglob("*"):
                        if file_path.is_file():
                            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_mtime < cutoff_date:
                                file_path.unlink()
                                cleaned_count += 1
                                self.logger.debug(f"Cleaned up old image: {file_path}")
            
            # Clean up old metadata files
            metadata_dir = self.base_directory / "metadata"
            if metadata_dir.exists():
                for file_path in metadata_dir.rglob("*.json"):
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_path.unlink()
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return cleaned_count
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['elapsed_time'] = time.time() - stats['start_time']
        
        # Add directory statistics
        stats['categories'] = {}
        for category in JewelryCategory:
            category_dir = self.base_directory / category.value
            if category_dir.exists():
                image_count = len(list(category_dir.rglob("*.jpg"))) + len(list(category_dir.rglob("*.png")))
                stats['categories'][category.value] = image_count
        
        return stats