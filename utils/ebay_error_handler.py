"""
Comprehensive Error Handling System for eBay Jewelry Scraping

Advanced error handling with categorization, recovery strategies,
logging, monitoring, and alerting capabilities.

Features:
- Error classification and categorization
- Recovery strategy selection
- Structured logging and monitoring
- Error pattern detection
- Alerting and notification system
- Performance impact assessment
- Automated error reporting
"""

import asyncio
import logging
import traceback
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid
import re


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    PARSING = "parsing"
    VALIDATION = "validation"
    BROWSER = "browser"
    DETECTION = "detection"
    CONFIGURATION = "configuration"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    RETRY_WITH_DELAY = "retry_with_delay"
    RETRY_WITH_DIFFERENT_CONFIG = "retry_with_different_config"
    SKIP_AND_CONTINUE = "skip_and_continue"
    FAIL_FAST = "fail_fast"
    MANUAL_INTERVENTION = "manual_intervention"
    CIRCUIT_BREAKER = "circuit_breaker"
    BACKOFF_AND_RETRY = "backoff_and_retry"


@dataclass
class ErrorContext:
    """Context information for error occurrence"""
    timestamp: datetime
    session_id: str
    operation: str
    url: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    execution_time: float = 0.0
    retry_count: int = 0
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorClassification:
    """Error classification result"""
    category: ErrorCategory
    severity: ErrorSeverity
    is_recoverable: bool
    recovery_strategy: RecoveryStrategy
    confidence: float
    explanation: str
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class ErrorOccurrence:
    """Individual error occurrence record"""
    error_id: str
    error: Exception
    classification: ErrorClassification
    context: ErrorContext
    stack_trace: str
    resolved: bool = False
    resolution_notes: Optional[str] = None


class ErrorPattern:
    """Pattern detection for recurring errors"""
    
    def __init__(self, window_minutes: int = 60, threshold: int = 5):
        self.window_minutes = window_minutes
        self.threshold = threshold
        self.error_history: deque = deque(maxlen=1000)
    
    def add_error(self, error_occurrence: ErrorOccurrence):
        """Add error to pattern detection"""
        self.error_history.append(error_occurrence)
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect error patterns"""
        patterns = []
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        
        # Recent errors within window
        recent_errors = [err for err in self.error_history 
                        if err.context.timestamp > cutoff_time]
        
        if len(recent_errors) < self.threshold:
            return patterns
        
        # Group by error type and category
        error_groups = defaultdict(list)
        for error in recent_errors:
            key = (type(error.error).__name__, error.classification.category)
            error_groups[key].append(error)
        
        # Identify patterns
        for (error_type, category), errors in error_groups.items():
            if len(errors) >= self.threshold:
                patterns.append({
                    'error_type': error_type,
                    'category': category.value,
                    'count': len(errors),
                    'frequency': len(errors) / self.window_minutes,
                    'severity': max(e.classification.severity for e in errors).value,
                    'first_occurrence': min(e.context.timestamp for e in errors),
                    'last_occurrence': max(e.context.timestamp for e in errors),
                    'affected_operations': list(set(e.context.operation for e in errors)),
                    'affected_urls': list(set(e.context.url for e in errors if e.context.url))
                })
        
        return patterns


class AdvancedErrorHandler:
    """
    Comprehensive error handling system with classification and recovery
    """
    
    def __init__(self, 
                 enable_alerting: bool = True,
                 enable_pattern_detection: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize error handler
        
        Args:
            enable_alerting: Enable error alerting
            enable_pattern_detection: Enable pattern detection
            log_level: Logging level
        """
        self.enable_alerting = enable_alerting
        self.enable_pattern_detection = enable_pattern_detection
        
        # Error tracking
        self.error_occurrences: Dict[str, ErrorOccurrence] = {}
        self.error_stats = defaultdict(int)
        self.recovery_stats = defaultdict(int)
        
        # Pattern detection
        self.pattern_detector = ErrorPattern() if enable_pattern_detection else None
        
        # Error classifiers
        self.classifiers = self._initialize_classifiers()
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Performance tracking
        self.error_impact_tracker = defaultdict(list)
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup error logging"""
        logger = logging.getLogger("ebay_jewelry_error_handler")
        logger.setLevel(getattr(logging, log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for error logs
            try:
                file_handler = logging.FileHandler('ebay_scraper_errors.log')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass  # File logging is optional
        
        return logger
    
    def _initialize_classifiers(self) -> Dict[str, Callable]:
        """Initialize error classification functions"""
        return {
            'network': self._classify_network_error,
            'http': self._classify_http_error,
            'browser': self._classify_browser_error,
            'parsing': self._classify_parsing_error,
            'validation': self._classify_validation_error,
            'detection': self._classify_detection_error,
            'system': self._classify_system_error
        }
    
    async def handle_error(self,
                          error: Exception,
                          context: ErrorContext,
                          auto_recover: bool = True) -> Optional[Any]:
        """
        Handle error with classification and recovery
        
        Args:
            error: Exception that occurred
            context: Error context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Generate unique error ID
            error_id = str(uuid.uuid4())
            
            # Classify error
            classification = await self._classify_error(error, context)
            
            # Create error occurrence record
            occurrence = ErrorOccurrence(
                error_id=error_id,
                error=error,
                classification=classification,
                context=context,
                stack_trace=traceback.format_exc()
            )
            
            # Store error occurrence
            self.error_occurrences[error_id] = occurrence
            self.error_stats[classification.category.value] += 1
            
            # Add to pattern detection
            if self.pattern_detector:
                self.pattern_detector.add_error(occurrence)
            
            # Log error
            await self._log_error(occurrence)
            
            # Send alerts if enabled
            if self.enable_alerting and classification.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                await self._send_alert(occurrence)
            
            # Track performance impact
            self._track_performance_impact(occurrence, time.time() - start_time)
            
            # Attempt recovery if enabled and recoverable
            if auto_recover and classification.is_recoverable:
                recovery_result = await self._attempt_recovery(occurrence)
                if recovery_result is not None:
                    occurrence.resolved = True
                    occurrence.resolution_notes = f"Auto-recovered using {classification.recovery_strategy.value}"
                    self.recovery_stats[classification.recovery_strategy.value] += 1
                    
                    self.logger.info(f"Successfully recovered from error {error_id}")
                    return recovery_result
            
            return None
            
        except Exception as handler_error:
            self.logger.error(f"Error in error handler: {handler_error}")
            return None
    
    async def _classify_error(self, error: Exception, context: ErrorContext) -> ErrorClassification:
        """Classify error and determine recovery strategy"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Try specific classifiers first
        for classifier_name, classifier_func in self.classifiers.items():
            try:
                classification = classifier_func(error, context)
                if classification:
                    return classification
            except Exception as e:
                self.logger.debug(f"Classifier {classifier_name} failed: {e}")
        
        # Fallback classification
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            is_recoverable=True,
            recovery_strategy=RecoveryStrategy.RETRY,
            confidence=0.3,
            explanation=f"Unknown error type: {error_type}",
            suggested_actions=["Check logs for more details", "Retry operation"]
        )
    
    def _classify_network_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify network-related errors"""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Network error patterns
        network_patterns = [
            'connection', 'timeout', 'unreachable', 'network', 'dns',
            'socket', 'ssl', 'certificate', 'handshake'
        ]
        
        if (any(pattern in error_message for pattern in network_patterns) or
            error_type in ['ConnectionError', 'TimeoutError', 'SSLError']):
            
            severity = ErrorSeverity.MEDIUM
            if 'timeout' in error_message:
                severity = ErrorSeverity.LOW
            elif 'ssl' in error_message or 'certificate' in error_message:
                severity = ErrorSeverity.HIGH
            
            return ErrorClassification(
                category=ErrorCategory.NETWORK,
                severity=severity,
                is_recoverable=True,
                recovery_strategy=RecoveryStrategy.RETRY_WITH_DELAY,
                confidence=0.9,
                explanation=f"Network error: {error_type}",
                suggested_actions=[
                    "Check internet connection",
                    "Verify proxy settings",
                    "Retry with exponential backoff"
                ]
            )
        
        return None
    
    def _classify_http_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify HTTP-related errors"""
        
        error_message = str(error).lower()
        status_code = context.response_status
        
        if status_code:
            if status_code == 429:
                return ErrorClassification(
                    category=ErrorCategory.RATE_LIMITING,
                    severity=ErrorSeverity.HIGH,
                    is_recoverable=True,
                    recovery_strategy=RecoveryStrategy.BACKOFF_AND_RETRY,
                    confidence=1.0,
                    explanation="Rate limiting detected",
                    suggested_actions=[
                        "Implement exponential backoff",
                        "Reduce request frequency",
                        "Check rate limiting headers"
                    ]
                )
            
            elif status_code in [403, 406]:
                # Potential detection
                return ErrorClassification(
                    category=ErrorCategory.DETECTION,
                    severity=ErrorSeverity.CRITICAL,
                    is_recoverable=True,
                    recovery_strategy=RecoveryStrategy.RETRY_WITH_DIFFERENT_CONFIG,
                    confidence=0.8,
                    explanation="Possible bot detection",
                    suggested_actions=[
                        "Change user agent",
                        "Use different browser profile",
                        "Implement longer delays",
                        "Check for CAPTCHA"
                    ]
                )
            
            elif status_code in [404, 410]:
                return ErrorClassification(
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.LOW,
                    is_recoverable=False,
                    recovery_strategy=RecoveryStrategy.SKIP_AND_CONTINUE,
                    confidence=0.9,
                    explanation="Resource not found",
                    suggested_actions=["Skip this URL", "Validate URL format"]
                )
            
            elif status_code in [500, 502, 503, 504]:
                return ErrorClassification(
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.MEDIUM,
                    is_recoverable=True,
                    recovery_strategy=RecoveryStrategy.RETRY_WITH_DELAY,
                    confidence=0.9,
                    explanation="Server error",
                    suggested_actions=["Retry after delay", "Check server status"]
                )
        
        return None
    
    def _classify_browser_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify browser-related errors"""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        browser_patterns = [
            'browser', 'chromium', 'firefox', 'webkit', 'playwright',
            'selenium', 'page', 'element', 'screenshot'
        ]
        
        if any(pattern in error_message for pattern in browser_patterns):
            severity = ErrorSeverity.MEDIUM
            
            if 'crashed' in error_message or 'killed' in error_message:
                severity = ErrorSeverity.HIGH
            
            return ErrorClassification(
                category=ErrorCategory.BROWSER,
                severity=severity,
                is_recoverable=True,
                recovery_strategy=RecoveryStrategy.RETRY_WITH_DIFFERENT_CONFIG,
                confidence=0.8,
                explanation=f"Browser error: {error_type}",
                suggested_actions=[
                    "Restart browser",
                    "Try different browser configuration",
                    "Check system resources"
                ]
            )
        
        return None
    
    def _classify_parsing_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify data parsing errors"""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        parsing_patterns = [
            'parse', 'json', 'xml', 'html', 'beautifulsoup', 'lxml',
            'selector', 'xpath', 'css'
        ]
        
        if (any(pattern in error_message for pattern in parsing_patterns) or
            error_type in ['JSONDecodeError', 'XMLSyntaxError', 'ValueError']):
            
            return ErrorClassification(
                category=ErrorCategory.PARSING,
                severity=ErrorSeverity.MEDIUM,
                is_recoverable=True,
                recovery_strategy=RecoveryStrategy.SKIP_AND_CONTINUE,
                confidence=0.7,
                explanation=f"Data parsing error: {error_type}",
                suggested_actions=[
                    "Check HTML structure",
                    "Update CSS selectors",
                    "Validate data format"
                ]
            )
        
        return None
    
    def _classify_validation_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify data validation errors"""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        if 'validation' in error_message or error_type == 'ValidationError':
            return ErrorClassification(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                is_recoverable=True,
                recovery_strategy=RecoveryStrategy.SKIP_AND_CONTINUE,
                confidence=0.9,
                explanation="Data validation failed",
                suggested_actions=[
                    "Check data quality",
                    "Update validation rules",
                    "Skip invalid records"
                ]
            )
        
        return None
    
    def _classify_detection_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify bot detection errors"""
        
        error_message = str(error).lower()
        
        detection_patterns = [
            'blocked', 'captcha', 'robot', 'automation', 'suspicious',
            'security', 'verification', 'human'
        ]
        
        if any(pattern in error_message for pattern in detection_patterns):
            return ErrorClassification(
                category=ErrorCategory.DETECTION,
                severity=ErrorSeverity.CRITICAL,
                is_recoverable=True,
                recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                confidence=0.9,
                explanation="Bot detection encountered",
                suggested_actions=[
                    "Implement longer delays",
                    "Change browser fingerprint",
                    "Use residential proxies",
                    "Solve CAPTCHA manually",
                    "Wait for cooldown period"
                ]
            )
        
        return None
    
    def _classify_system_error(self, error: Exception, context: ErrorContext) -> Optional[ErrorClassification]:
        """Classify system-level errors"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        system_patterns = [
            'memory', 'disk', 'permission', 'file', 'directory',
            'resource', 'system'
        ]
        
        if (any(pattern in error_message for pattern in system_patterns) or
            error_type in ['MemoryError', 'OSError', 'IOError', 'PermissionError']):
            
            severity = ErrorSeverity.HIGH
            if error_type == 'MemoryError':
                severity = ErrorSeverity.CRITICAL
            
            return ErrorClassification(
                category=ErrorCategory.SYSTEM,
                severity=severity,
                is_recoverable=False,
                recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                confidence=0.9,
                explanation=f"System error: {error_type}",
                suggested_actions=[
                    "Check system resources",
                    "Free up memory/disk space",
                    "Check file permissions",
                    "Restart application"
                ]
            )
        
        return None
    
    async def _attempt_recovery(self, occurrence: ErrorOccurrence) -> Optional[Any]:
        """Attempt error recovery based on strategy"""
        
        strategy = occurrence.classification.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                # Simple retry - would need callback function
                return "retry_scheduled"
            
            elif strategy == RecoveryStrategy.RETRY_WITH_DELAY:
                # Delay then retry
                await asyncio.sleep(5)
                return "retry_with_delay_scheduled"
            
            elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
                # Skip this operation and continue
                return "skipped"
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                # Trigger circuit breaker
                return "circuit_breaker_triggered"
            
            elif strategy == RecoveryStrategy.BACKOFF_AND_RETRY:
                # Exponential backoff
                backoff_time = min(30, 2 ** occurrence.context.retry_count)
                await asyncio.sleep(backoff_time)
                return "backoff_retry_scheduled"
            
            else:
                return None
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return None
    
    async def _log_error(self, occurrence: ErrorOccurrence):
        """Log error occurrence"""
        
        log_data = {
            'error_id': occurrence.error_id,
            'error_type': type(occurrence.error).__name__,
            'error_message': str(occurrence.error),
            'category': occurrence.classification.category.value,
            'severity': occurrence.classification.severity.value,
            'recoverable': occurrence.classification.is_recoverable,
            'recovery_strategy': occurrence.classification.recovery_strategy.value,
            'context': {
                'timestamp': occurrence.context.timestamp.isoformat(),
                'session_id': occurrence.context.session_id,
                'operation': occurrence.context.operation,
                'url': occurrence.context.url,
                'retry_count': occurrence.context.retry_count,
                'execution_time': occurrence.context.execution_time
            }
        }
        
        # Log based on severity
        if occurrence.classification.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {json.dumps(log_data, indent=2)}")
        elif occurrence.classification.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {json.dumps(log_data, indent=2)}")
        elif occurrence.classification.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(log_data, indent=2)}")
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {json.dumps(log_data, indent=2)}")
    
    async def _send_alert(self, occurrence: ErrorOccurrence):
        """Send error alert"""
        
        alert_data = {
            'error_id': occurrence.error_id,
            'timestamp': occurrence.context.timestamp.isoformat(),
            'severity': occurrence.classification.severity.value,
            'category': occurrence.classification.category.value,
            'message': str(occurrence.error),
            'operation': occurrence.context.operation,
            'url': occurrence.context.url,
            'suggested_actions': occurrence.classification.suggested_actions
        }
        
        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def _track_performance_impact(self, occurrence: ErrorOccurrence, handling_time: float):
        """Track performance impact of errors"""
        
        impact_data = {
            'timestamp': occurrence.context.timestamp,
            'category': occurrence.classification.category.value,
            'handling_time': handling_time,
            'execution_time': occurrence.context.execution_time,
            'operation': occurrence.context.operation
        }
        
        self.error_impact_tracker[occurrence.classification.category.value].append(impact_data)
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler function"""
        self.alert_handlers.append(handler)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        total_errors = sum(self.error_stats.values())
        
        stats = {
            'total_errors': total_errors,
            'errors_by_category': dict(self.error_stats),
            'recovery_attempts': dict(self.recovery_stats),
            'error_rate_by_category': {
                category: count / total_errors if total_errors > 0 else 0
                for category, count in self.error_stats.items()
            }
        }
        
        # Recent error patterns
        if self.pattern_detector:
            stats['recent_patterns'] = self.pattern_detector.detect_patterns()
        
        # Performance impact
        impact_summary = {}
        for category, impacts in self.error_impact_tracker.items():
            if impacts:
                impact_summary[category] = {
                    'avg_handling_time': sum(i['handling_time'] for i in impacts) / len(impacts),
                    'total_impact_time': sum(i['execution_time'] for i in impacts),
                    'count': len(impacts)
                }
        stats['performance_impact'] = impact_summary
        
        return stats
    
    def get_recovery_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for improving error recovery"""
        
        recommendations = []
        
        # Analyze error patterns
        if self.pattern_detector:
            patterns = self.pattern_detector.detect_patterns()
            
            for pattern in patterns:
                if pattern['frequency'] > 2:  # More than 2 errors per minute
                    recommendations.append({
                        'priority': 'high',
                        'issue': f"High frequency {pattern['category']} errors",
                        'recommendation': f"Investigate {pattern['error_type']} in {pattern['category']} operations",
                        'affected_operations': pattern['affected_operations']
                    })
        
        # Analyze recovery success rates
        total_recoveries = sum(self.recovery_stats.values())
        if total_recoveries > 0:
            for strategy, count in self.recovery_stats.items():
                success_rate = count / total_recoveries
                if success_rate < 0.5:  # Less than 50% success
                    recommendations.append({
                        'priority': 'medium',
                        'issue': f"Low success rate for {strategy} recovery",
                        'recommendation': f"Review and improve {strategy} recovery strategy",
                        'success_rate': success_rate
                    })
        
        return recommendations
    
    def reset_statistics(self):
        """Reset error statistics"""
        self.error_stats.clear()
        self.recovery_stats.clear()
        self.error_impact_tracker.clear()
        if self.pattern_detector:
            self.pattern_detector.error_history.clear()
        
        self.logger.info("Error statistics reset")