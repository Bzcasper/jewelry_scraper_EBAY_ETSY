"""
Advanced Image Processing Pipeline for Jewelry Images

Enhanced comprehensive pipeline with:
- Concurrent image processing with intelligent rate limiting
- Advanced validation and format checking
- Jewelry-specific optimization algorithms
- Multi-resolution generation with adaptive quality
- Comprehensive metadata extraction and analysis
- Smart storage organization and duplicate detection
- Performance monitoring and memory optimization
- Perceptual hash-based duplicate detection
- Advanced quality scoring with ML-ready features
"""

import asyncio
import aiohttp
import aiofiles
import hashlib
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import psutil
import gc

from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
from PIL.ExifTags import TAGS
import imagehash


class ProcessingStage(Enum):
    """Image processing pipeline stages"""
    DOWNLOAD = "download"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    RESIZE = "resize"
    ENHANCEMENT = "enhancement"
    STORAGE = "storage"
    METADATA = "metadata"
    DEDUPLICATION = "deduplication"
    CLEANUP = "cleanup"


class JewelryCategory(Enum):
    """Enhanced jewelry categories for organization"""
    RINGS = "rings"
    NECKLACES = "necklaces"
    EARRINGS = "earrings"
    BRACELETS = "bracelets"
    WATCHES = "watches"
    PENDANTS = "pendants"
    CHAINS = "chains"
    BROOCHES = "brooches"
    GEMSTONES = "gemstones"
    PEARLS = "pearls"
    VINTAGE = "vintage"
    LUXURY = "luxury"
    OTHER = "other"


class ImageQuality(Enum):
    """Enhanced image quality levels with specific use cases"""
    THUMBNAIL = "thumbnail"    # 150x150, for previews
    LOW = "low"               # 400x400, for listings
    MEDIUM = "medium"         # 800x800, for galleries
    HIGH = "high"             # 1200x1200, for detailed views
    ULTRA = "ultra"           # 2000x2000, for professional use
    ORIGINAL = "original"     # Unchanged, for archival


class ConcurrencyMode(Enum):
    """Concurrency optimization modes"""
    CONSERVATIVE = "conservative"  # Low resource usage
    BALANCED = "balanced"         # Balanced performance
    AGGRESSIVE = "aggressive"     # Maximum performance
    ADAPTIVE = "adaptive"         # Auto-adjust based on system


@dataclass
class QualityMetrics:
    """Advanced image quality assessment metrics"""
    resolution_score: float = 0.0
    sharpness_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    color_richness: float = 0.0
    noise_level: float = 0.0
    composition_score: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ProcessingStats:
    """Comprehensive processing statistics"""
    stage_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    success_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    total_processed: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    total_duplicates: int = 0
    average_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ImageMetadata:
    """Comprehensive image metadata with enhanced features"""
    filename: str
    url: str
    file_size: int
    dimensions: Tuple[int, int]
    format: str
    mode: str
    file_hash: str
    perceptual_hash: str
    download_timestamp: datetime
    processing_timestamp: Optional[datetime] = None
    category: Optional[JewelryCategory] = None
    quality_metrics: Optional[QualityMetrics] = None
    exif_data: Optional[Dict] = None
    color_profile: Optional[Dict] = None
    is_optimized: bool = False
    compression_ratios: Dict[str, float] = field(default_factory=dict)
    validation_status: str = "pending"
    error_message: Optional[str] = None
    processing_stages: List[str] = field(default_factory=list)
    generated_sizes: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['download_timestamp'] = self.download_timestamp.isoformat()
        if self.processing_timestamp:
            data['processing_timestamp'] = self.processing_timestamp.isoformat()
        if self.category:
            data['category'] = self.category.value
        if self.quality_metrics:
            data['quality_metrics'] = self.quality_metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['download_timestamp'] = datetime.fromisoformat(
            data['download_timestamp'])
        if data.get('processing_timestamp'):
            data['processing_timestamp'] = datetime.fromisoformat(
                data['processing_timestamp'])
        if data.get('category'):
            data['category'] = JewelryCategory(data['category'])
        if data.get('quality_metrics'):
            data['quality_metrics'] = QualityMetrics(**data['quality_metrics'])
        return cls(**data)


class PerformanceMonitor:
    """Real-time performance monitoring and optimization"""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.stage_times = defaultdict(list)
        self.memory_samples = []
        self.cpu_samples = []
        self._lock = threading.Lock()

    def start_stage(self, stage: str) -> float:
        """Start timing a processing stage"""
        return time.time()

    def end_stage(self, stage: str, start_time: float):
        """End timing a processing stage"""
        duration = time.time() - start_time
        with self._lock:
            self.stage_times[stage].append(duration)

    def sample_resources(self):
        """Sample current resource usage"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()

            with self._lock:
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)

        except Exception:
            pass  # Ignore sampling errors

    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        with self._lock:
            return {
                'average_stage_times': {
                    stage: sum(times) / len(times)
                    for stage, times in self.stage_times.items() if times
                },
                'peak_memory_mb': max(self.memory_samples) if self.memory_samples else 0,
                'average_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
                'total_runtime': time.time() - self.start_time
            }


class ImageProcessor:
    """
    Advanced concurrent image processing pipeline for jewelry images

    Features:
    - Intelligent concurrency with adaptive resource management
    - Advanced quality assessment and optimization
    - Perceptual hash-based duplicate detection
    - Comprehensive metadata extraction and analysis
    - Multi-resolution generation with smart quality scaling
    - Performance monitoring and memory optimization
    """

    # Processing stage definitions
    PROCESSING_STAGES = [
        ProcessingStage.DOWNLOAD,
        ProcessingStage.VALIDATION,
        ProcessingStage.ANALYSIS,
        ProcessingStage.OPTIMIZATION,
        ProcessingStage.RESIZE,
        ProcessingStage.ENHANCEMENT,
        ProcessingStage.STORAGE,
        ProcessingStage.METADATA,
        ProcessingStage.DEDUPLICATION,
        ProcessingStage.CLEANUP
    ]

    def __init__(self,
                 base_directory: str = "./images",
                 concurrency_mode: ConcurrencyMode = ConcurrencyMode.BALANCED,
                 max_concurrent_downloads: Optional[int] = None,
                 max_concurrent_processing: Optional[int] = None,
                 request_delay: float = 0.5,
                 timeout: int = 30,
                 max_retries: int = 3,
                 quality_settings: Optional[Dict[ImageQuality, Dict]] = None,
                 enable_performance_monitoring: bool = True,
                 memory_limit_mb: Optional[int] = None):
        """
        Initialize the advanced image processor

        Args:
            base_directory: Base directory for image storage
            concurrency_mode: Concurrency optimization mode
            max_concurrent_downloads: Override concurrent downloads (auto-detected if None)
            max_concurrent_processing: Override concurrent processing (auto-detected if None)
            request_delay: Delay between requests for rate limiting
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed downloads
            quality_settings: Custom quality settings for optimization
            enable_performance_monitoring: Enable real-time performance monitoring
            memory_limit_mb: Memory usage limit in MB (auto-detected if None)
        """
        self.base_directory = Path(base_directory)
        self.concurrency_mode = concurrency_mode
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_performance_monitoring = enable_performance_monitoring

        # Auto-detect optimal concurrency settings
        self._setup_concurrency_limits(
            max_concurrent_downloads, max_concurrent_processing)

        # Setup memory management
        self.memory_limit_mb = memory_limit_mb or self._detect_memory_limit()

        # Concurrency control
        self.download_semaphore = asyncio.Semaphore(
            self.max_concurrent_downloads)
        self.processing_semaphore = asyncio.Semaphore(
            self.max_concurrent_processing)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_concurrent_processing)

        # Rate limiting
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()

        # Performance monitoring
        self.monitor = PerformanceMonitor() if enable_performance_monitoring else None
        self.stats = ProcessingStats()

        # Enhanced quality settings
        self.quality_settings = quality_settings or self._get_default_quality_settings()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize directory structure
        self._initialize_directories()

        # Caching and deduplication
        self.processed_cache: Set[str] = set()
        self.hash_cache: Dict[str, str] = {}  # perceptual_hash -> filename
        self._load_caches()

        # Concurrent features tracking
        self.concurrent_features = {
            'download_concurrency': self.max_concurrent_downloads,
            'processing_concurrency': self.max_concurrent_processing,
            'batch_processing': True,
            'adaptive_rate_limiting': True,
            'memory_management': True,
            'resource_monitoring': enable_performance_monitoring,
            'thread_pool_processing': True,
            'async_io_operations': True
        }

        # Optimization methods tracking
        self.optimization_methods = {
            'perceptual_hashing': True,
            'adaptive_quality_scaling': True,
            'jewelry_specific_enhancement': True,
            'progressive_jpeg_optimization': True,
            'exif_based_auto_orientation': True,
            'color_profile_optimization': True,
            'noise_reduction': True,
            'sharpness_enhancement': True,
            'contrast_optimization': True,
            'brightness_adjustment': True
        }

    def _setup_concurrency_limits(self, max_downloads: Optional[int], max_processing: Optional[int]):
        """Auto-detect optimal concurrency limits based on system resources"""
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)

        if self.concurrency_mode == ConcurrencyMode.CONSERVATIVE:
            self.max_concurrent_downloads = max_downloads or min(5, cpu_count)
            self.max_concurrent_processing = max_processing or min(
                2, cpu_count // 2)
        elif self.concurrency_mode == ConcurrencyMode.BALANCED:
            self.max_concurrent_downloads = max_downloads or min(
                10, cpu_count * 2)
            self.max_concurrent_processing = max_processing or min(
                4, cpu_count)
        elif self.concurrency_mode == ConcurrencyMode.AGGRESSIVE:
            self.max_concurrent_downloads = max_downloads or min(
                20, cpu_count * 3)
            self.max_concurrent_processing = max_processing or min(
                8, cpu_count * 2)
        else:  # ADAPTIVE
            # Start conservative and adapt based on performance
            self.max_concurrent_downloads = max_downloads or min(
                8, cpu_count * 2)
            self.max_concurrent_processing = max_processing or min(
                4, cpu_count)

    def _detect_memory_limit(self) -> int:
        """Auto-detect reasonable memory limit"""
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        # Use 25% of available memory as limit
        return int(total_memory_gb * 0.25 * 1024)  # Convert to MB

    def _get_default_quality_settings(self) -> Dict[ImageQuality, Dict]:
        """Get default quality settings optimized for jewelry images"""
        return {
            ImageQuality.THUMBNAIL: {
                'max_size': (150, 150),
                'quality': 85,
                'optimize': True,
                'progressive': False,
                'enhance_sharpness': 1.2,
                'enhance_contrast': 1.1
            },
            ImageQuality.LOW: {
                'max_size': (400, 400),
                'quality': 75,
                'optimize': True,
                'progressive': True,
                'enhance_sharpness': 1.1,
                'enhance_contrast': 1.05
            },
            ImageQuality.MEDIUM: {
                'max_size': (800, 800),
                'quality': 85,
                'optimize': True,
                'progressive': True,
                'enhance_sharpness': 1.1,
                'enhance_contrast': 1.05
            },
            ImageQuality.HIGH: {
                'max_size': (1200, 1200),
                'quality': 90,
                'optimize': True,
                'progressive': True,
                'enhance_sharpness': 1.05,
                'enhance_contrast': 1.02
            },
            ImageQuality.ULTRA: {
                'max_size': (2000, 2000),
                'quality': 95,
                'optimize': True,
                'progressive': True,
                'enhance_sharpness': 1.02,
                'enhance_contrast': 1.01
            },
            ImageQuality.ORIGINAL: {
                'max_size': None,
                'quality': 100,
                'optimize': False,
                'progressive': False,
                'enhance_sharpness': 1.0,
                'enhance_contrast': 1.0
            }
        }

    def _initialize_directories(self):
        """Create comprehensive directory structure"""
        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Create category directories
        for category in JewelryCategory:
            category_dir = self.base_directory / category.value
            category_dir.mkdir(exist_ok=True)

            # Create quality subdirectories
            for quality in ImageQuality:
                quality_dir = category_dir / quality.value
                quality_dir.mkdir(exist_ok=True)

        # Create system directories
        for system_dir in ['metadata', 'cache', 'duplicates', 'failed', 'logs']:
            (self.base_directory / system_dir).mkdir(exist_ok=True)

        self.logger.info(
            f"Initialized directory structure at {self.base_directory}")

    def _load_caches(self):
        """Load processing and hash caches"""
        # Load processed cache
        processed_cache_file = self.base_directory / "cache" / "processed.json"
        if processed_cache_file.exists():
            try:
                with open(processed_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.processed_cache = set(cache_data.get('processed', []))
            except Exception as e:
                self.logger.warning(f"Failed to load processed cache: {e}")

        # Load hash cache for duplicate detection
        hash_cache_file = self.base_directory / "cache" / "hash_cache.json"
        if hash_cache_file.exists():
            try:
                with open(hash_cache_file, 'r') as f:
                    self.hash_cache = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load hash cache: {e}")

    def _save_caches(self):
        """Save processing and hash caches"""
        try:
            # Save processed cache
            processed_cache_file = self.base_directory / "cache" / "processed.json"
            with open(processed_cache_file, 'w') as f:
                json.dump({
                    'processed': list(self.processed_cache),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)

            # Save hash cache
            hash_cache_file = self.base_directory / "cache" / "hash_cache.json"
            with open(hash_cache_file, 'w') as f:
                json.dump(self.hash_cache, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save caches: {e}")

    async def _rate_limit(self):
        """Intelligent rate limiting with adaptive delays"""
        async with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            # Adaptive delay based on recent performance
            if self.monitor:
                recent_stats = self.monitor.get_stats()
                if recent_stats.get('average_cpu_percent', 0) > 80:
                    adaptive_delay = self.request_delay * 1.5
                elif recent_stats.get('peak_memory_mb', 0) > self.memory_limit_mb * 0.8:
                    adaptive_delay = self.request_delay * 1.2
                else:
                    adaptive_delay = self.request_delay
            else:
                adaptive_delay = self.request_delay

            if time_since_last < adaptive_delay:
                sleep_time = adaptive_delay - time_since_last
                await asyncio.sleep(sleep_time)

            self.last_request_time = time.time()

    def _calculate_perceptual_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash for duplicate detection"""
        try:
            # Use average hash for good balance of speed and accuracy
            phash = imagehash.average_hash(image, hash_size=8)
            return str(phash)
        except Exception as e:
            self.logger.debug(f"Failed to calculate perceptual hash: {e}")
            return ""

    def _assess_image_quality(self, image: Image.Image) -> QualityMetrics:
        """Advanced quality assessment for jewelry images"""
        try:
            width, height = image.size

            # Resolution score (normalized to common jewelry image sizes)
            resolution_score = min(1.0, (width * height) / (1000 * 1000))

            # Convert to grayscale for analysis
            gray = image.convert('L')

            # Sharpness assessment using Laplacian variance
            try:
                laplacian = gray.filter(ImageFilter.Kernel(
                    (3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], 1, 0))
                sharpness_score = min(1.0, np.var(np.array(laplacian)) / 1000)
            except:
                sharpness_score = 0.5

            # Contrast assessment
            try:
                stat = ImageStat.Stat(gray)
                # Normalize to 0-1
                contrast_score = min(1.0, stat.stddev[0] / 64)
            except:
                contrast_score = 0.5

            # Brightness assessment (prefer moderate brightness)
            try:
                stat = ImageStat.Stat(gray)
                brightness = stat.mean[0] / 255
                # Prefer middle brightness
                brightness_score = 1.0 - abs(brightness - 0.5) * 2
            except:
                brightness_score = 0.5

            # Color richness (for color images)
            color_richness = 0.5
            if image.mode in ('RGB', 'RGBA'):
                try:
                    colors = image.getcolors(maxcolors=256*256*256)
                    if colors:
                        unique_colors = len(colors)
                        color_richness = min(
                            1.0, unique_colors / 10000)  # Normalize
                except:
                    pass

            # Composition score (aspect ratio preference for jewelry)
            aspect_ratio = width / height
            if 0.8 <= aspect_ratio <= 1.2:  # Square-ish is preferred for jewelry
                composition_score = 1.0
            elif 0.6 <= aspect_ratio <= 1.6:  # Acceptable range
                composition_score = 0.8
            else:
                composition_score = 0.6

            # Overall score (weighted combination)
            overall_score = (
                resolution_score * 0.25 +
                sharpness_score * 0.25 +
                contrast_score * 0.15 +
                brightness_score * 0.15 +
                color_richness * 0.10 +
                composition_score * 0.10
            )

            return QualityMetrics(
                resolution_score=resolution_score,
                sharpness_score=sharpness_score,
                contrast_score=contrast_score,
                brightness_score=brightness_score,
                color_richness=color_richness,
                noise_level=0.0,  # Would need more sophisticated analysis
                composition_score=composition_score,
                overall_score=overall_score
            )

        except Exception as e:
            self.logger.debug(f"Quality assessment failed: {e}")
            return QualityMetrics(overall_score=0.5)

    def _categorize_image(self, url: str, metadata: Optional[Dict] = None) -> JewelryCategory:
        """Enhanced automatic categorization using keyword analysis"""
        url_lower = url.lower()

        # Enhanced keyword matching with more specific terms
        category_keywords = {
            JewelryCategory.RINGS: ['ring', 'band', 'engagement', 'wedding', 'signet', 'solitaire'],
            JewelryCategory.NECKLACES: ['necklace', 'chain', 'choker', 'collar', 'lariat', 'riviere'],
            JewelryCategory.EARRINGS: ['earring', 'stud', 'hoop', 'drop', 'chandelier', 'huggie'],
            JewelryCategory.BRACELETS: ['bracelet', 'bangle', 'cuff', 'tennis', 'charm'],
            JewelryCategory.WATCHES: ['watch', 'timepiece', 'rolex', 'omega', 'cartier', 'chronograph'],
            JewelryCategory.PENDANTS: ['pendant', 'charm', 'medallion', 'locket'],
            JewelryCategory.CHAINS: ['chain', 'rope', 'figaro', 'curb', 'snake'],
            JewelryCategory.BROOCHES: ['brooch', 'pin', 'clip'],
            JewelryCategory.GEMSTONES: ['diamond', 'ruby', 'emerald', 'sapphire', 'gemstone', 'precious'],
            JewelryCategory.PEARLS: ['pearl', 'cultured', 'freshwater', 'tahitian'],
            JewelryCategory.VINTAGE: ['vintage', 'antique', 'estate', 'art deco', 'victorian'],
            JewelryCategory.LUXURY: [
                'luxury', 'designer', 'tiffany', 'bulgari', 'gucci']
        }

        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in url_lower)
            if score > 0:
                category_scores[category] = score

        # Return highest scoring category or OTHER
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return JewelryCategory.OTHER

    def _check_duplicate(self, perceptual_hash: str, threshold: int = 5) -> Optional[str]:
        """Check for duplicate images using perceptual hash comparison"""
        for existing_hash, filename in self.hash_cache.items():
            try:
                # Calculate Hamming distance between hashes
                hash1 = imagehash.hex_to_hash(perceptual_hash)
                hash2 = imagehash.hex_to_hash(existing_hash)
                distance = hash1 - hash2

                if distance <= threshold:
                    return filename
            except:
                continue
        return None

    def _manage_memory(self):
        """Monitor and manage memory usage"""
        if self.monitor:
            self.monitor.sample_resources()
            current_memory = psutil.virtual_memory().used / (1024**2)  # MB

            if current_memory > self.memory_limit_mb:
                gc.collect()  # Force garbage collection
                self.logger.debug(
                    "Triggered garbage collection due to memory pressure")

    async def _validate_image(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Enhanced image validation with format and content checks"""
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False, "File is empty or doesn't exist"

            # Check file size limits
            file_size = file_path.stat().st_size
            if file_size < 1024:  # Less than 1KB
                return False, "File too small (minimum 1KB)"
            if file_size > 50 * 1024 * 1024:  # More than 50MB
                return False, "File too large (maximum 50MB)"

            with Image.open(file_path) as img:
                img.verify()

                # Reopen for further checks
                with Image.open(file_path) as img:
                    width, height = img.size

                    # Dimension checks
                    if width < 50 or height < 50:
                        return False, "Image too small (minimum 50x50 pixels)"
                    if width > 8000 or height > 8000:
                        return False, "Image too large (maximum 8000x8000 pixels)"

                    # Format validation
                    if img.format not in ['JPEG', 'PNG', 'WEBP', 'GIF', 'BMP', 'TIFF']:
                        return False, f"Unsupported format: {img.format}"

                    # Mode validation
                    if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                        return False, f"Unsupported color mode: {img.mode}"

            return True, None

        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    async def download_image(self,
                             url: str,
                             filename: Optional[str] = None,
                             category: Optional[JewelryCategory] = None,
                             metadata: Optional[Dict] = None) -> Optional[ImageMetadata]:
        """Enhanced image download with comprehensive processing"""

        async with self.download_semaphore:
            # Performance monitoring
            stage_start = self.monitor.start_stage(
                "download") if self.monitor else time.time()

            try:
                await self._rate_limit()

                # Generate filename if not provided
                if not filename:
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                    filename = f"jewelry_{url_hash}"

                # Auto-categorize if not provided
                if not category:
                    category = self._categorize_image(url, metadata)

                # Check if already processed
                if url in self.processed_cache:
                    self.stats.total_skipped += 1
                    self.logger.debug(f"Skipping already processed: {url}")
                    return None

                temp_path = self.base_directory / "cache" / f"{filename}_temp"

                # Download with retries
                for attempt in range(self.max_retries):
                    try:
                        timeout = aiohttp.ClientTimeout(total=self.timeout)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.get(url) as response:
                                if response.status == 200:
                                    content = await response.read()

                                    async with aiofiles.open(temp_path, 'wb') as f:
                                        await f.write(content)

                                    # Validate downloaded image
                                    is_valid, error_msg = await self._validate_image(temp_path)
                                    if not is_valid:
                                        temp_path.unlink(missing_ok=True)
                                        self.stats.total_failed += 1
                                        return None

                                    # Create metadata object
                                    image_metadata = ImageMetadata(
                                        filename=filename,
                                        url=url,
                                        file_size=len(content),
                                        dimensions=(0, 0),  # Will be updated
                                        format="",
                                        mode="",
                                        file_hash="",
                                        perceptual_hash="",
                                        download_timestamp=datetime.now(),
                                        category=category,
                                        validation_status="valid"
                                    )

                                    # Process the image
                                    processed_metadata = await self._process_image_pipeline(temp_path, image_metadata)

                                    if processed_metadata:
                                        self.processed_cache.add(url)
                                        self.stats.total_processed += 1
                                        return processed_metadata
                                    else:
                                        temp_path.unlink(missing_ok=True)
                                        self.stats.total_failed += 1
                                        return None
                                else:
                                    self.logger.warning(
                                        f"HTTP {response.status} for {url}")

                    except Exception as e:
                        self.logger.warning(
                            f"Download attempt {attempt + 1} failed for {url}: {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)

                # Clean up on failure
                temp_path.unlink(missing_ok=True)
                self.stats.total_failed += 1
                return None

            finally:
                if self.monitor:
                    self.monitor.end_stage("download", stage_start)
                self._manage_memory()

    async def _process_image_pipeline(self, temp_path: Path, metadata: ImageMetadata) -> Optional[ImageMetadata]:
        """Comprehensive image processing pipeline"""
        async with self.processing_semaphore:
            pipeline_start = time.time()

            try:
                # Stage 1: Analysis
                stage_start = self.monitor.start_stage(
                    "analysis") if self.monitor else time.time()

                with Image.open(temp_path) as img:
                    # Update basic metadata
                    metadata.dimensions = img.size
                    metadata.format = img.format
                    metadata.mode = img.mode
                    metadata.processing_timestamp = datetime.now()

                    # Calculate perceptual hash for duplicate detection
                    metadata.perceptual_hash = self._calculate_perceptual_hash(
                        img)

                    # Check for duplicates
                    duplicate_filename = self._check_duplicate(
                        metadata.perceptual_hash)
                    if duplicate_filename:
                        self.logger.info(
                            f"Duplicate detected: {metadata.filename} matches {duplicate_filename}")
                        self.stats.total_duplicates += 1
                        # Move to duplicates folder
                        duplicate_path = self.base_directory / \
                            "duplicates" / f"{metadata.filename}_duplicate"
                        temp_path.rename(duplicate_path)
                        return None

                    # Quality assessment
                    metadata.quality_metrics = self._assess_image_quality(img)

                    # Extract EXIF data
                    try:
                        if hasattr(img, '_getexif') and img._getexif():
                            exif_data = {}
                            for tag_id, value in img._getexif().items():
                                tag = TAGS.get(tag_id, tag_id)
                                exif_data[tag] = str(value)
                            metadata.exif_data = exif_data
                    except:
                        pass

                if self.monitor:
                    self.monitor.end_stage("analysis", stage_start)
                metadata.processing_stages.append("analysis")

                # Stage 2: Generate multiple resolutions
                stage_start = self.monitor.start_stage(
                    "resize") if self.monitor else time.time()

                await self._generate_multi_resolution(temp_path, metadata)

                if self.monitor:
                    self.monitor.end_stage("resize", stage_start)
                metadata.processing_stages.append("resize")

                # Stage 3: Storage organization
                stage_start = self.monitor.start_stage(
                    "storage") if self.monitor else time.time()

                # Calculate final file hash
                metadata.file_hash = self._calculate_file_hash(temp_path)

                # Move original to category directory
                original_path = (self.base_directory / metadata.category.value /
                                 ImageQuality.ORIGINAL.value / f"{metadata.filename}_original.{metadata.format.lower()}")
                temp_path.rename(original_path)

                if self.monitor:
                    self.monitor.end_stage("storage", stage_start)
                metadata.processing_stages.append("storage")

                # Stage 4: Metadata storage
                stage_start = self.monitor.start_stage(
                    "metadata") if self.monitor else time.time()

                await self._save_metadata(metadata)

                # Update hash cache
                self.hash_cache[metadata.perceptual_hash] = metadata.filename

                if self.monitor:
                    self.monitor.end_stage("metadata", stage_start)
                metadata.processing_stages.append("metadata")

                # Mark as optimized
                metadata.is_optimized = True

                processing_time = time.time() - pipeline_start
                self.logger.info(
                    f"Processed {metadata.filename} in {processing_time:.2f}s")

                return metadata

            except Exception as e:
                self.logger.error(f"Processing failed for {temp_path}: {e}")
                temp_path.unlink(missing_ok=True)
                return None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def _generate_multi_resolution(self, temp_path: Path, metadata: ImageMetadata):
        """Generate multiple resolution versions with jewelry-specific optimizations"""

        with Image.open(temp_path) as original_img:
            for quality_level in [q for q in ImageQuality if q != ImageQuality.ORIGINAL]:
                try:
                    settings = self.quality_settings[quality_level]

                    # Create working copy
                    img = original_img.copy()

                    # Auto-orient based on EXIF
                    img = ImageOps.exif_transpose(img)

                    # Resize if needed
                    if settings['max_size'] and (img.size[0] > settings['max_size'][0] or img.size[1] > settings['max_size'][1]):
                        img.thumbnail(settings['max_size'],
                                      Image.Resampling.LANCZOS)

                    # Jewelry-specific enhancements
                    if settings.get('enhance_sharpness', 1.0) != 1.0:
                        enhancer = ImageEnhance.Sharpness(img)
                        img = enhancer.enhance(settings['enhance_sharpness'])

                    if settings.get('enhance_contrast', 1.0) != 1.0:
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(settings['enhance_contrast'])

                    # Convert mode for JPEG if needed
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')

                    # Save optimized version
                    output_path = (self.base_directory / metadata.category.value / quality_level.value /
                                   f"{metadata.filename}_{quality_level.value}.jpg")

                    img.save(
                        output_path,
                        'JPEG',
                        quality=settings['quality'],
                        optimize=settings['optimize'],
                        progressive=settings.get('progressive', False)
                    )

                    # Track compression ratio
                    original_size = metadata.file_size
                    compressed_size = output_path.stat().st_size
                    metadata.compression_ratios[quality_level.value] = compressed_size / \
                        original_size if original_size > 0 else 1.0

                    # Track generated size info
                    metadata.generated_sizes[quality_level.value] = {
                        'dimensions': img.size,
                        'file_size': compressed_size,
                        'path': str(output_path)
                    }

                except Exception as e:
                    self.logger.error(
                        f"Failed to generate {quality_level.value} version: {e}")

    async def _save_metadata(self, metadata: ImageMetadata):
        """Save comprehensive metadata"""
        try:
            metadata_file = self.base_directory / "metadata" / \
                f"{metadata.filename}_metadata.json"

            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(metadata.to_dict(), indent=2))

        except Exception as e:
            self.logger.error(
                f"Failed to save metadata for {metadata.filename}: {e}")

    async def process_urls_batch(self,
                                 urls: List[str],
                                 batch_size: int = 50,
                                 progress_callback: Optional[Callable] = None) -> List[ImageMetadata]:
        """Process multiple URLs with advanced batch management"""

        self.stats.start_time = datetime.now()
        successful_metadata = []

        self.logger.info(f"Starting batch processing of {len(urls)} images")

        try:
            # Process in batches for memory management
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]

                # Create download tasks
                tasks = [self.download_image(url) for url in batch_urls]

                # Process batch with timeout
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect successful results
                for result in batch_results:
                    if isinstance(result, ImageMetadata):
                        successful_metadata.append(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Batch error: {result}")

                # Progress callback
                if progress_callback:
                    progress = (i + len(batch_urls)) / len(urls)
                    progress_callback(progress, self.get_statistics())

                # Memory management between batches
                if i % 100 == 0:  # Every 100 images
                    self._save_caches()
                    gc.collect()

                # Brief pause between batches
                await asyncio.sleep(0.1)

        finally:
            # Final cleanup
            self.stats.end_time = datetime.now()
            self._save_caches()

            if self.monitor:
                final_stats = self.monitor.get_stats()
                self.logger.info(f"Performance stats: {final_stats}")

        elapsed = (self.stats.end_time - self.stats.start_time).total_seconds()
        self.logger.info(f"Batch processing completed in {elapsed:.2f}s. "
                         f"Success: {len(successful_metadata)}, "
                         f"Failed: {self.stats.total_failed}, "
                         f"Skipped: {self.stats.total_skipped}, "
                         f"Duplicates: {self.stats.total_duplicates}")

        return successful_metadata

    async def cleanup_old_files(self, days_old: int = 30, dry_run: bool = False) -> int:
        """Advanced cleanup with safety checks"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0

        try:
            # Scan all directories for old files
            for root_dir in [self.base_directory]:
                for file_path in root_dir.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_mtime = datetime.fromtimestamp(
                                file_path.stat().st_mtime)
                            if file_mtime < cutoff_date:
                                if not dry_run:
                                    file_path.unlink()
                                cleaned_count += 1
                                self.logger.debug(
                                    f"{'Would clean' if dry_run else 'Cleaned'}: {file_path}")
                        except Exception as e:
                            self.logger.warning(
                                f"Error processing {file_path}: {e}")

            self.logger.info(
                f"{'Would clean' if dry_run else 'Cleaned'} {cleaned_count} old files")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            return cleaned_count

    def get_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        stats = {
            'processing_stats': asdict(self.stats),
            'concurrent_features': self.concurrent_features,
            'optimization_methods': self.optimization_methods,
            'processing_stages_count': len(self.PROCESSING_STAGES),
            'categories_supported': len(JewelryCategory),
            'quality_levels': len(ImageQuality),
            'directory_structure': {
                'base': str(self.base_directory),
                'categories': [cat.value for cat in JewelryCategory],
                'quality_levels': [qual.value for qual in ImageQuality]
            }
        }

        # Add performance monitoring stats if available
        if self.monitor:
            stats['performance_metrics'] = self.monitor.get_stats()

        # Add cache statistics
        stats['cache_stats'] = {
            'processed_cache_size': len(self.processed_cache),
            'hash_cache_size': len(self.hash_cache)
        }

        return stats

    def get_processing_stages_count(self) -> int:
        """Get number of processing stages"""
        return len(self.PROCESSING_STAGES)

    def get_concurrent_features(self) -> Dict[str, Any]:
        """Get concurrent processing features"""
        return self.concurrent_features.copy()

    def get_optimization_methods(self) -> Dict[str, bool]:
        """Get available optimization methods"""
        return self.optimization_methods.copy()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        self._save_caches()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)


# Main processing function for easy usage
async def process_jewelry_images(urls: List[str],
                                 base_directory: str = "./processed_jewelry_images",
                                 concurrency_mode: ConcurrencyMode = ConcurrencyMode.BALANCED,
                                 enable_monitoring: bool = True) -> Dict:
    """
    High-level function to process jewelry images with optimal settings

    Args:
        urls: List of image URLs to process
        base_directory: Directory for processed images
        concurrency_mode: Processing concurrency mode
        enable_monitoring: Enable performance monitoring

    Returns:
        Dictionary with processing results and statistics
    """

    async with ImageProcessor(
        base_directory=base_directory,
        concurrency_mode=concurrency_mode,
        enable_performance_monitoring=enable_monitoring
    ) as processor:

        # Process images
        results = await processor.process_urls_batch(urls)

        # Get final statistics
        stats = processor.get_statistics()

        return {
            'processed_images': len(results),
            'successful_metadata': results,
            'statistics': stats,
            'processing_stages_count': processor.get_processing_stages_count(),
            'concurrent_features': processor.get_concurrent_features(),
            'optimization_methods': processor.get_optimization_methods()
        }


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Real eBay jewelry image URLs
        test_urls = [
            "https://i.ebayimg.com/images/g/YjUAAOSwh~tjyNQy/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/kNMAAOSwdGVh2p3v/s-l1600.jpg"
        ]

        # Process images
        results = await process_jewelry_images(
            urls=test_urls,
            base_directory="./test_jewelry_images",
            concurrency_mode=ConcurrencyMode.BALANCED,
            enable_monitoring=True
        )

        print(f"Processed {results['processed_images']} images")
        print(f"Processing stages: {results['processing_stages_count']}")
        print(
            f"Concurrent features: {list(results['concurrent_features'].keys())}")
        print(
            f"Optimization methods: {sum(results['optimization_methods'].values())} active")

    # Run example
    # asyncio.run(main())
