"""
Advanced Backup Manager for Jewelry Database
Provides comprehensive backup and recovery capabilities with multiple strategies.
"""

import os
import shutil
import sqlite3
import gzip
import tarfile
import zipfile
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import threading
import time

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class BackupStrategy(str, Enum):
    """Backup strategy types"""
    FULL = "full"           # Complete database backup
    INCREMENTAL = "incremental"  # Only changes since last backup
    DIFFERENTIAL = "differential"  # Changes since last full backup
    SCHEMA_ONLY = "schema_only"   # Structure only, no data
    DATA_ONLY = "data_only"       # Data only, no structure


class BackupFormat(str, Enum):
    """Backup file formats"""
    SQL_DUMP = "sql_dump"
    SQLITE_COPY = "sqlite_copy"
    CSV_EXPORT = "csv_export"
    JSON_EXPORT = "json_export"


class CompressionType(str, Enum):
    """Compression types for backups"""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar_gz"


class StorageLocation(str, Enum):
    """Storage location types"""
    LOCAL = "local"
    AWS_S3 = "aws_s3"
    GOOGLE_CLOUD = "google_cloud"
    FTP = "ftp"
    SFTP = "sftp"


class BackupStatus(str, Enum):
    """Backup operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupConfig:
    """Backup configuration settings"""
    
    # Basic settings
    strategy: BackupStrategy = BackupStrategy.FULL
    format: BackupFormat = BackupFormat.SQLITE_COPY
    compression: CompressionType = CompressionType.GZIP
    
    # Storage settings
    storage_location: StorageLocation = StorageLocation.LOCAL
    local_backup_dir: str = "backups"
    max_local_backups: int = 10
    
    # Cloud storage settings (if applicable)
    aws_bucket: Optional[str] = None
    aws_region: Optional[str] = None
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    
    gcs_bucket: Optional[str] = None
    gcs_credentials_path: Optional[str] = None
    
    # Retention settings
    retention_days: int = 30
    auto_cleanup: bool = True
    
    # Performance settings
    chunk_size: int = 8192
    max_concurrent_uploads: int = 3
    
    # Verification settings
    verify_backup: bool = True
    checksum_algorithm: str = "sha256"
    
    # Notification settings
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_emails: List[str] = field(default_factory=list)
    
    # Custom settings
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupInfo:
    """Information about a backup"""
    
    backup_id: str
    filename: str
    file_path: str
    strategy: BackupStrategy
    format: BackupFormat
    compression: CompressionType
    storage_location: StorageLocation
    
    # Size and timing
    file_size_bytes: int
    compressed_size_bytes: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    duration_seconds: Optional[float] = None
    
    # Verification
    checksum: Optional[str] = None
    checksum_algorithm: Optional[str] = None
    is_verified: bool = False
    
    # Database info at backup time
    record_counts: Dict[str, int] = field(default_factory=dict)
    database_size_mb: Optional[float] = None
    
    # Status and metadata
    status: BackupStatus = BackupStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupManager:
    """
    Advanced backup manager with multiple strategies and storage options
    """
    
    def __init__(self, database_manager: DatabaseManager, config: Optional[BackupConfig] = None):
        """
        Initialize backup manager
        
        Args:
            database_manager: DatabaseManager instance
            config: Backup configuration (uses defaults if None)
        """
        self.db_manager = database_manager
        self.config = config or BackupConfig()
        
        # Ensure backup directory exists
        Path(self.config.local_backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize backup tracking
        self._backup_history: List[BackupInfo] = []
        self._active_backups: Dict[str, threading.Thread] = {}
        
        # Initialize cloud storage clients
        self._aws_client = None
        self._gcs_client = None
        self._init_cloud_clients()
        
        logger.info(f"BackupManager initialized with {self.config.strategy} strategy")
    
    def _init_cloud_clients(self):
        """Initialize cloud storage clients"""
        
        if self.config.storage_location == StorageLocation.AWS_S3 and AWS_AVAILABLE:
            try:
                self._aws_client = boto3.client(
                    's3',
                    region_name=self.config.aws_region,
                    aws_access_key_id=self.config.aws_access_key,
                    aws_secret_access_key=self.config.aws_secret_key
                )
                logger.info("AWS S3 client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AWS S3 client: {e}")
        
        if self.config.storage_location == StorageLocation.GOOGLE_CLOUD and GCS_AVAILABLE:
            try:
                if self.config.gcs_credentials_path:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.gcs_credentials_path
                self._gcs_client = gcs.Client()
                logger.info("Google Cloud Storage client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}")
    
    def create_backup(self, 
                     strategy: Optional[BackupStrategy] = None,
                     custom_name: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> BackupInfo:
        """
        Create a database backup
        
        Args:
            strategy: Backup strategy (uses config default if None)
            custom_name: Custom backup name
            metadata: Additional metadata
            
        Returns:
            BackupInfo object with backup details
        """
        
        strategy = strategy or self.config.strategy
        backup_id = self._generate_backup_id()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name:
            filename = f"{custom_name}_{timestamp}"
        else:
            filename = f"jewelry_backup_{strategy.value}_{timestamp}"
        
        # Add format extension
        if self.config.format == BackupFormat.SQL_DUMP:
            filename += ".sql"
        elif self.config.format == BackupFormat.SQLITE_COPY:
            filename += ".db"
        elif self.config.format == BackupFormat.CSV_EXPORT:
            filename += ".csv"
        elif self.config.format == BackupFormat.JSON_EXPORT:
            filename += ".json"
        
        # Add compression extension
        if self.config.compression == CompressionType.GZIP:
            filename += ".gz"
        elif self.config.compression == CompressionType.ZIP:
            filename += ".zip"
        elif self.config.compression == CompressionType.TAR_GZ:
            filename += ".tar.gz"
        
        # Create backup info
        backup_info = BackupInfo(
            backup_id=backup_id,
            filename=filename,
            file_path=str(Path(self.config.local_backup_dir) / filename),
            strategy=strategy,
            format=self.config.format,
            compression=self.config.compression,
            storage_location=self.config.storage_location,
            metadata=metadata or {}
        )
        
        # Start backup in background thread
        backup_thread = threading.Thread(
            target=self._perform_backup,
            args=(backup_info,),
            name=f"backup_{backup_id}"
        )
        
        self._active_backups[backup_id] = backup_thread
        backup_thread.start()
        
        # Add to history
        self._backup_history.append(backup_info)
        
        logger.info(f"Started backup {backup_id} with {strategy} strategy")
        
        return backup_info
    
    def _perform_backup(self, backup_info: BackupInfo):
        """Perform the actual backup operation"""
        
        start_time = time.time()
        backup_info.status = BackupStatus.RUNNING
        
        try:
            logger.info(f"Performing backup {backup_info.backup_id}")
            
            # Get database statistics before backup
            db_stats = self.db_manager.get_enhanced_database_stats()
            backup_info.record_counts = {
                'listings': db_stats.total_listings,
                'images': db_stats.total_images,
                'specifications': db_stats.total_specifications,
                'sessions': db_stats.total_sessions
            }
            backup_info.database_size_mb = db_stats.storage_size_mb
            
            # Perform backup based on strategy and format
            if backup_info.strategy == BackupStrategy.FULL:
                self._create_full_backup(backup_info)
            elif backup_info.strategy == BackupStrategy.INCREMENTAL:
                self._create_incremental_backup(backup_info)
            elif backup_info.strategy == BackupStrategy.DIFFERENTIAL:
                self._create_differential_backup(backup_info)
            elif backup_info.strategy == BackupStrategy.SCHEMA_ONLY:
                self._create_schema_backup(backup_info)
            elif backup_info.strategy == BackupStrategy.DATA_ONLY:
                self._create_data_backup(backup_info)
            
            # Apply compression if specified
            if self.config.compression != CompressionType.NONE:
                self._compress_backup(backup_info)
            
            # Calculate file size
            backup_info.file_size_bytes = Path(backup_info.file_path).stat().st_size
            
            # Calculate checksum for verification
            if self.config.verify_backup:
                backup_info.checksum = self._calculate_checksum(backup_info.file_path)
                backup_info.checksum_algorithm = self.config.checksum_algorithm
                backup_info.is_verified = True
            
            # Upload to cloud storage if configured
            if self.config.storage_location != StorageLocation.LOCAL:
                self._upload_to_cloud(backup_info)
            
            # Calculate duration
            backup_info.duration_seconds = time.time() - start_time
            backup_info.status = BackupStatus.COMPLETED
            
            logger.info(f"Backup {backup_info.backup_id} completed successfully in {backup_info.duration_seconds:.2f}s")
            
            # Cleanup old backups if configured
            if self.config.auto_cleanup:
                self._cleanup_old_backups()
            
            # Send notification if configured
            if self.config.notify_on_success:
                self._send_notification(backup_info, success=True)
        
        except Exception as e:
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.duration_seconds = time.time() - start_time
            
            logger.error(f"Backup {backup_info.backup_id} failed: {e}")
            
            # Send failure notification
            if self.config.notify_on_failure:
                self._send_notification(backup_info, success=False)
            
            raise
        
        finally:
            # Remove from active backups
            self._active_backups.pop(backup_info.backup_id, None)
    
    def _create_full_backup(self, backup_info: BackupInfo):
        """Create a full database backup"""
        
        if backup_info.format == BackupFormat.SQLITE_COPY:
            # Simple file copy for SQLite
            shutil.copy2(self.db_manager.db_path, backup_info.file_path)
        
        elif backup_info.format == BackupFormat.SQL_DUMP:
            self._create_sql_dump(backup_info.file_path)
        
        elif backup_info.format == BackupFormat.JSON_EXPORT:
            self._create_json_export(backup_info.file_path)
        
        elif backup_info.format == BackupFormat.CSV_EXPORT:
            self._create_csv_export(backup_info.file_path)
    
    def _create_incremental_backup(self, backup_info: BackupInfo):
        """Create an incremental backup (changes since last backup)"""
        
        # Find last backup time
        last_backup_time = self._get_last_backup_time()
        
        if not last_backup_time:
            # No previous backup, perform full backup
            logger.info("No previous backup found, performing full backup")
            self._create_full_backup(backup_info)
            return
        
        # Export only records modified since last backup
        self._create_filtered_backup(
            backup_info.file_path,
            last_backup_time,
            backup_info.format
        )
    
    def _create_differential_backup(self, backup_info: BackupInfo):
        """Create a differential backup (changes since last full backup)"""
        
        # Find last full backup time
        last_full_backup_time = self._get_last_full_backup_time()
        
        if not last_full_backup_time:
            # No previous full backup, perform full backup
            logger.info("No previous full backup found, performing full backup")
            self._create_full_backup(backup_info)
            return
        
        # Export only records modified since last full backup
        self._create_filtered_backup(
            backup_info.file_path,
            last_full_backup_time,
            backup_info.format
        )
    
    def _create_schema_backup(self, backup_info: BackupInfo):
        """Create a schema-only backup"""
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            with open(backup_info.file_path, 'w') as f:
                # Export only the schema
                for line in conn.iterdump():
                    if line.startswith('CREATE') or line.startswith('PRAGMA'):
                        f.write(line + '\n')
    
    def _create_data_backup(self, backup_info: BackupInfo):
        """Create a data-only backup"""
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            with open(backup_info.file_path, 'w') as f:
                # Export only INSERT statements
                for line in conn.iterdump():
                    if line.startswith('INSERT'):
                        f.write(line + '\n')
    
    def _create_sql_dump(self, file_path: str):
        """Create SQL dump of database"""
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            with open(file_path, 'w') as f:
                for line in conn.iterdump():
                    f.write(line + '\n')
    
    def _create_json_export(self, file_path: str):
        """Create JSON export of database"""
        
        export_data = {}
        
        tables = ['jewelry_listings', 'jewelry_images', 'jewelry_specifications', 'scraping_sessions']
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            for table in tables:
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                
                # Convert rows to dictionaries
                columns = [description[0] for description in cursor.description]
                export_data[table] = [
                    dict(zip(columns, row)) for row in rows
                ]
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _create_csv_export(self, file_path: str):
        """Create CSV export of main listings table"""
        
        import csv
        
        with self.db_manager.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jewelry_listings")
            
            with open(file_path, 'w', newline='') as csvfile:
                # Write header
                columns = [description[0] for description in cursor.description]
                writer = csv.writer(csvfile)
                writer.writerow(columns)
                
                # Write data
                for row in cursor.fetchall():
                    writer.writerow(row)
    
    def _create_filtered_backup(self, file_path: str, since_time: datetime, format: BackupFormat):
        """Create backup with time-based filtering"""
        
        since_iso = since_time.isoformat()
        
        if format == BackupFormat.JSON_EXPORT:
            export_data = {}
            tables = ['jewelry_listings', 'jewelry_images', 'jewelry_specifications', 'scraping_sessions']
            
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                for table in tables:
                    # Use different time columns for different tables
                    time_column = 'scraped_at' if table == 'jewelry_listings' else 'created_at'
                    
                    cursor.execute(f"""
                        SELECT * FROM {table} 
                        WHERE {time_column} > ?
                    """, (since_iso,))
                    
                    rows = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    export_data[table] = [
                        dict(zip(columns, row)) for row in rows
                    ]
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == BackupFormat.CSV_EXPORT:
            import csv
            
            with self.db_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM jewelry_listings 
                    WHERE scraped_at > ?
                """, (since_iso,))
                
                with open(file_path, 'w', newline='') as csvfile:
                    columns = [description[0] for description in cursor.description]
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    
                    for row in cursor.fetchall():
                        writer.writerow(row)
    
    def _compress_backup(self, backup_info: BackupInfo):
        """Apply compression to backup file"""
        
        original_path = backup_info.file_path
        
        if self.config.compression == CompressionType.GZIP:
            compressed_path = original_path + '.gz'
            
            with open(original_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original and update path
            os.remove(original_path)
            backup_info.file_path = compressed_path
            backup_info.filename = Path(compressed_path).name
        
        elif self.config.compression == CompressionType.ZIP:
            compressed_path = original_path + '.zip'
            
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(original_path, Path(original_path).name)
            
            os.remove(original_path)
            backup_info.file_path = compressed_path
            backup_info.filename = Path(compressed_path).name
        
        elif self.config.compression == CompressionType.TAR_GZ:
            compressed_path = original_path + '.tar.gz'
            
            with tarfile.open(compressed_path, 'w:gz') as tar:
                tar.add(original_path, arcname=Path(original_path).name)
            
            os.remove(original_path)
            backup_info.file_path = compressed_path
            backup_info.filename = Path(compressed_path).name
        
        # Update compressed size
        backup_info.compressed_size_bytes = Path(backup_info.file_path).stat().st_size
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum for verification"""
        
        hash_algo = hashlib.new(self.config.checksum_algorithm)
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(self.config.chunk_size):
                hash_algo.update(chunk)
        
        return hash_algo.hexdigest()
    
    def _upload_to_cloud(self, backup_info: BackupInfo):
        """Upload backup to cloud storage"""
        
        if backup_info.storage_location == StorageLocation.AWS_S3:
            self._upload_to_s3(backup_info)
        elif backup_info.storage_location == StorageLocation.GOOGLE_CLOUD:
            self._upload_to_gcs(backup_info)
    
    def _upload_to_s3(self, backup_info: BackupInfo):
        """Upload backup to AWS S3"""
        
        if not self._aws_client:
            raise RuntimeError("AWS S3 client not initialized")
        
        try:
            with open(backup_info.file_path, 'rb') as f:
                self._aws_client.upload_fileobj(
                    f,
                    self.config.aws_bucket,
                    backup_info.filename,
                    ExtraArgs={
                        'Metadata': {
                            'backup_id': backup_info.backup_id,
                            'strategy': backup_info.strategy.value,
                            'created_at': backup_info.created_at.isoformat(),
                            'checksum': backup_info.checksum or ''
                        }
                    }
                )
            
            logger.info(f"Uploaded backup {backup_info.backup_id} to S3")
        
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise
    
    def _upload_to_gcs(self, backup_info: BackupInfo):
        """Upload backup to Google Cloud Storage"""
        
        if not self._gcs_client:
            raise RuntimeError("Google Cloud Storage client not initialized")
        
        try:
            bucket = self._gcs_client.bucket(self.config.gcs_bucket)
            blob = bucket.blob(backup_info.filename)
            
            # Set metadata
            blob.metadata = {
                'backup_id': backup_info.backup_id,
                'strategy': backup_info.strategy.value,
                'created_at': backup_info.created_at.isoformat(),
                'checksum': backup_info.checksum or ''
            }
            
            blob.upload_from_filename(backup_info.file_path)
            
            logger.info(f"Uploaded backup {backup_info.backup_id} to GCS")
        
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise
    
    def _get_last_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last backup"""
        
        completed_backups = [
            b for b in self._backup_history 
            if b.status == BackupStatus.COMPLETED
        ]
        
        if not completed_backups:
            return None
        
        return max(b.created_at for b in completed_backups)
    
    def _get_last_full_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last full backup"""
        
        full_backups = [
            b for b in self._backup_history 
            if b.status == BackupStatus.COMPLETED and b.strategy == BackupStrategy.FULL
        ]
        
        if not full_backups:
            return None
        
        return max(b.created_at for b in full_backups)
    
    def _cleanup_old_backups(self):
        """Clean up old local backup files"""
        
        try:
            # Get all backup files in local directory
            backup_dir = Path(self.config.local_backup_dir)
            backup_files = []
            
            for file_path in backup_dir.iterdir():
                if file_path.is_file():
                    backup_files.append((file_path, file_path.stat().st_mtime))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove files beyond max count
            if len(backup_files) > self.config.max_local_backups:
                for file_path, _ in backup_files[self.config.max_local_backups:]:
                    file_path.unlink()
                    logger.info(f"Removed old backup file: {file_path}")
            
            # Remove files older than retention period
            cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
            for file_path, mtime in backup_files:
                if mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Removed expired backup file: {file_path}")
        
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")
    
    def _send_notification(self, backup_info: BackupInfo, success: bool):
        """Send backup notification (placeholder for email/webhook integration)"""
        
        # This would integrate with actual notification services
        status = "successful" if success else "failed"
        message = f"Backup {backup_info.backup_id} {status}"
        
        if not success and backup_info.error_message:
            message += f": {backup_info.error_message}"
        
        logger.info(f"Notification: {message}")
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}_{hash(time.time()) % 10000:04d}"
    
    # === PUBLIC METHODS ===
    
    def restore_backup(self, backup_info: BackupInfo, target_path: Optional[str] = None) -> bool:
        """
        Restore database from backup
        
        Args:
            backup_info: Backup to restore from
            target_path: Target database path (uses current if None)
            
        Returns:
            bool: Success status
        """
        
        target_path = target_path or str(self.db_manager.db_path)
        
        try:
            logger.info(f"Restoring backup {backup_info.backup_id} to {target_path}")
            
            # Download from cloud if necessary
            local_backup_path = backup_info.file_path
            if backup_info.storage_location != StorageLocation.LOCAL:
                local_backup_path = self._download_backup(backup_info)
            
            # Verify backup if checksum available
            if backup_info.checksum and self.config.verify_backup:
                current_checksum = self._calculate_checksum(local_backup_path)
                if current_checksum != backup_info.checksum:
                    raise ValueError("Backup checksum verification failed")
            
            # Decompress if necessary
            if backup_info.compression != CompressionType.NONE:
                local_backup_path = self._decompress_backup(local_backup_path, backup_info.compression)
            
            # Restore based on format
            if backup_info.format == BackupFormat.SQLITE_COPY:
                shutil.copy2(local_backup_path, target_path)
            
            elif backup_info.format == BackupFormat.SQL_DUMP:
                self._restore_from_sql_dump(local_backup_path, target_path)
            
            elif backup_info.format == BackupFormat.JSON_EXPORT:
                self._restore_from_json(local_backup_path, target_path)
            
            else:
                raise ValueError(f"Restore not supported for format: {backup_info.format}")
            
            logger.info(f"Successfully restored backup {backup_info.backup_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_info.backup_id}: {e}")
            return False
    
    def _download_backup(self, backup_info: BackupInfo) -> str:
        """Download backup from cloud storage"""
        
        local_path = str(Path(self.config.local_backup_dir) / f"restore_{backup_info.filename}")
        
        if backup_info.storage_location == StorageLocation.AWS_S3:
            self._aws_client.download_file(
                self.config.aws_bucket,
                backup_info.filename,
                local_path
            )
        
        elif backup_info.storage_location == StorageLocation.GOOGLE_CLOUD:
            bucket = self._gcs_client.bucket(self.config.gcs_bucket)
            blob = bucket.blob(backup_info.filename)
            blob.download_to_filename(local_path)
        
        return local_path
    
    def _decompress_backup(self, file_path: str, compression: CompressionType) -> str:
        """Decompress backup file"""
        
        if compression == CompressionType.GZIP:
            decompressed_path = file_path.replace('.gz', '')
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return decompressed_path
        
        elif compression == CompressionType.ZIP:
            with zipfile.ZipFile(file_path, 'r') as zipf:
                extracted = zipf.extractall(Path(file_path).parent)
                return str(Path(file_path).parent / zipf.namelist()[0])
        
        elif compression == CompressionType.TAR_GZ:
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(Path(file_path).parent)
                return str(Path(file_path).parent / tar.getnames()[0])
        
        return file_path
    
    def _restore_from_sql_dump(self, dump_path: str, target_path: str):
        """Restore database from SQL dump"""
        
        with sqlite3.connect(target_path) as conn:
            with open(dump_path, 'r') as f:
                conn.executescript(f.read())
    
    def _restore_from_json(self, json_path: str, target_path: str):
        """Restore database from JSON export"""
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # This would require recreating the database schema and inserting data
        # Implementation depends on the JSON structure
        raise NotImplementedError("JSON restore not yet implemented")
    
    def list_backups(self, 
                    strategy: Optional[BackupStrategy] = None,
                    status: Optional[BackupStatus] = None,
                    limit: Optional[int] = None) -> List[BackupInfo]:
        """
        List available backups
        
        Args:
            strategy: Filter by backup strategy
            status: Filter by backup status
            limit: Maximum number of results
            
        Returns:
            List of BackupInfo objects
        """
        
        backups = self._backup_history.copy()
        
        # Apply filters
        if strategy:
            backups = [b for b in backups if b.strategy == strategy]
        
        if status:
            backups = [b for b in backups if b.status == status]
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        
        # Apply limit
        if limit:
            backups = backups[:limit]
        
        return backups
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get information about a specific backup"""
        
        for backup in self._backup_history:
            if backup.backup_id == backup_id:
                return backup
        
        return None
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            bool: Success status
        """
        
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            return False
        
        try:
            # Delete local file if exists
            if Path(backup_info.file_path).exists():
                Path(backup_info.file_path).unlink()
            
            # Delete from cloud storage if applicable
            if backup_info.storage_location == StorageLocation.AWS_S3:
                self._aws_client.delete_object(
                    Bucket=self.config.aws_bucket,
                    Key=backup_info.filename
                )
            
            elif backup_info.storage_location == StorageLocation.GOOGLE_CLOUD:
                bucket = self._gcs_client.bucket(self.config.gcs_bucket)
                blob = bucket.blob(backup_info.filename)
                blob.delete()
            
            # Remove from history
            self._backup_history = [
                b for b in self._backup_history 
                if b.backup_id != backup_id
            ]
            
            logger.info(f"Deleted backup {backup_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics"""
        
        total_backups = len(self._backup_history)
        completed_backups = [b for b in self._backup_history if b.status == BackupStatus.COMPLETED]
        failed_backups = [b for b in self._backup_history if b.status == BackupStatus.FAILED]
        
        total_size = sum(b.file_size_bytes for b in completed_backups if b.file_size_bytes)
        
        return {
            'total_backups': total_backups,
            'completed_backups': len(completed_backups),
            'failed_backups': len(failed_backups),
            'active_backups': len(self._active_backups),
            'total_size_mb': total_size / (1024 * 1024) if total_size else 0,
            'latest_backup': max(b.created_at for b in completed_backups) if completed_backups else None,
            'success_rate': len(completed_backups) / total_backups * 100 if total_backups else 0
        }
    
    def schedule_automatic_backup(self, interval_hours: int = 24):
        """Schedule automatic backups (simplified implementation)"""
        
        def backup_scheduler():
            while True:
                time.sleep(interval_hours * 3600)
                try:
                    self.create_backup()
                    logger.info("Automatic backup completed")
                except Exception as e:
                    logger.error(f"Automatic backup failed: {e}")
        
        scheduler_thread = threading.Thread(target=backup_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info(f"Scheduled automatic backups every {interval_hours} hours")


# Convenience functions for common backup scenarios
def create_daily_backup(backup_manager: BackupManager) -> BackupInfo:
    """Create a daily backup with standard settings"""
    
    return backup_manager.create_backup(
        strategy=BackupStrategy.FULL,
        custom_name="daily",
        metadata={'backup_type': 'scheduled_daily'}
    )


def create_pre_update_backup(backup_manager: BackupManager) -> BackupInfo:
    """Create a backup before system updates"""
    
    return backup_manager.create_backup(
        strategy=BackupStrategy.FULL,
        custom_name="pre_update",
        metadata={'backup_type': 'pre_update', 'timestamp': datetime.now().isoformat()}
    )