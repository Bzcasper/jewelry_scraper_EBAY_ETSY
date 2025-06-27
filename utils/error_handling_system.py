"""
Robust Error Handling, Retry Logic, and Rate Limiting System
Comprehensive error management with intelligent recovery strategies
"""

import asyncio
import time
import logging
import traceback
import json
from typing import Dict, List, Optional, Any, Callable, Union, Type, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import hashlib
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classification"""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    BOT_DETECTION = "bot_detection"
    PARSING = "parsing"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    ADAPTIVE = "adaptive"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


@dataclass
class ErrorRecord:
    """Comprehensive error record with context"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    strategy: RetryStrategy
    max_attempts: int
    base_delay: float
    max_delay: float
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    backoff_multiplier: float = 1.0
    retry_on_status_codes: List[int] = None
    retry_on_exceptions: List[Type[Exception]] = None
    
    def __post_init__(self):
        if self.retry_on_status_codes is None:
            self.retry_on_status_codes = [429, 502, 503, 504, 520, 522, 524]
        if self.retry_on_exceptions is None:
            self.retry_on_exceptions = [
                ConnectionError, TimeoutError, asyncio.TimeoutError
            ]


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    strategy: RateLimitStrategy
    requests_per_second: float
    burst_size: int
    window_size: int = 60
    adaptive_increase_rate: float = 0.1
    adaptive_decrease_rate: float = 0.5
    min_requests_per_second: float = 0.1
    max_requests_per_second: float = 10.0


class TokenBucket:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket"""
        async with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_token(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens"""
        async with self._lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed_tokens = tokens - self.tokens
            wait_time = needed_tokens / self.rate
            return wait_time


class SlidingWindowCounter:
    """Sliding window rate limiter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def can_proceed(self) -> bool:
        """Check if request can proceed"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside window
            while self.requests and now - self.requests[0] > self.window_size:
                self.requests.popleft()
            
            return len(self.requests) < self.max_requests
    
    async def record_request(self):
        """Record a request"""
        async with self._lock:
            self.requests.append(time.time())


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    self.logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN call limit exceeded")
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            raise e
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info("Circuit breaker CLOSED - service recovered")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.half_open_calls = 0
                self.logger.warning("Circuit breaker back to OPEN from HALF_OPEN")


class ErrorClassifier:
    """Classify errors for appropriate handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Error patterns for classification
        self.classification_patterns = {
            ErrorCategory.NETWORK: [
                'connection', 'timeout', 'dns', 'socket', 'network',
                'unreachable', 'refused', 'reset', 'broken pipe'
            ],
            ErrorCategory.RATE_LIMIT: [
                'rate limit', 'throttl', '429', 'too many requests',
                'slow down', 'quota exceeded'
            ],
            ErrorCategory.BOT_DETECTION: [
                'captcha', 'robot', 'automated', 'suspicious',
                'blocked', 'access denied', 'forbidden', '403'
            ],
            ErrorCategory.PARSING: [
                'parse', 'parsing', 'json', 'xml', 'html', 'decode',
                'invalid format', 'malformed'
            ],
            ErrorCategory.VALIDATION: [
                'validation', 'invalid', 'required', 'constraint',
                'format', 'range', 'missing'
            ],
            ErrorCategory.AUTHENTICATION: [
                'auth', 'unauthorized', '401', 'login', 'credential',
                'token', 'expired'
            ],
            ErrorCategory.PERMISSION: [
                'permission', 'forbidden', '403', 'access denied',
                'not allowed', 'insufficient'
            ],
            ErrorCategory.RESOURCE: [
                'memory', 'disk', 'space', 'limit', 'quota',
                'resource', 'capacity'
            ],
            ErrorCategory.SYSTEM: [
                'system', 'os', 'file', 'process', 'thread',
                'deadlock', 'corruption'
            ]
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity"""
        
        error_text = f"{type(error).__name__} {str(error)}".lower()
        
        # Classify by category
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self.classification_patterns.items():
            if any(pattern in error_text for pattern in patterns):
                category = cat
                break
        
        # Determine severity
        severity = self._determine_severity(error, category, context)
        
        return category, severity
    
    def _determine_severity(self, error: Exception, category: ErrorCategory, 
                          context: Dict[str, Any] = None) -> ErrorSeverity:
        """Determine error severity"""
        
        # Critical errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        if category in [ErrorCategory.SYSTEM, ErrorCategory.RESOURCE]:
            return ErrorSeverity.HIGH
        
        # High severity
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.PERMISSION]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.BOT_DETECTION, ErrorCategory.RATE_LIMIT]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        if category in [ErrorCategory.PARSING, ErrorCategory.VALIDATION]:
            return ErrorSeverity.LOW
        
        # Network errors - depends on frequency
        if category == ErrorCategory.NETWORK:
            if context and context.get('retry_count', 0) > 3:
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.LOW
        
        return ErrorSeverity.LOW


class RetryHandler:
    """Advanced retry handler with multiple strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry(e, attempt):
                    self.logger.warning(f"Not retrying {type(e).__name__}: {e}")
                    raise e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_attempts} attempts failed")
        
        # All retries exhausted
        raise last_exception
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception should trigger retry"""
        
        # Check exception type
        if any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions):
            return True
        
        # Check HTTP status codes (if applicable)
        if hasattr(exception, 'status') or hasattr(exception, 'code'):
            status_code = getattr(exception, 'status', None) or getattr(exception, 'code', None)
            if status_code in self.config.retry_on_status_codes:
                return True
        
        # Check error message for rate limiting indicators
        error_msg = str(exception).lower()
        rate_limit_indicators = ['rate limit', 'throttl', '429', 'too many requests']
        if any(indicator in error_msg for indicator in rate_limit_indicators):
            return True
        
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on strategy"""
        
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1) * self.config.backoff_multiplier
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
            
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            # Adaptive strategy based on error type
            delay = self.config.base_delay * (1.5 ** attempt)
            
        else:  # IMMEDIATE or NO_RETRY
            delay = 0
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter and delay > 0:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on server response"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.current_rate = config.requests_per_second
        self.token_bucket = TokenBucket(self.current_rate, config.burst_size)
        self.sliding_window = SlidingWindowCounter(
            config.window_size, 
            int(config.requests_per_second * config.window_size)
        )
        
        # Adaptive behavior tracking
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 30  # seconds
        
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make request"""
        
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self.token_bucket.acquire()
        
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            can_proceed = await self.sliding_window.can_proceed()
            if can_proceed:
                await self.sliding_window.record_request()
            return can_proceed
        
        elif self.config.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._adaptive_acquire()
        
        else:  # FIXED_WINDOW
            return await self._fixed_window_acquire()
    
    async def _adaptive_acquire(self) -> bool:
        """Adaptive rate limiting with automatic adjustment"""
        
        # Check if we need to adjust rate
        await self._maybe_adjust_rate()
        
        # Use token bucket with current rate
        return await self.token_bucket.acquire()
    
    async def _maybe_adjust_rate(self):
        """Adjust rate based on recent performance"""
        
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        async with self._lock:
            total_requests = self.success_count + self.error_count
            
            if total_requests > 0:
                error_rate = self.error_count / total_requests
                
                if error_rate > 0.1:  # More than 10% errors
                    # Decrease rate
                    new_rate = self.current_rate * (1 - self.config.adaptive_decrease_rate)
                    new_rate = max(new_rate, self.config.min_requests_per_second)
                    
                    if new_rate != self.current_rate:
                        self.logger.info(f"Decreasing rate from {self.current_rate:.2f} to {new_rate:.2f}")
                        self.current_rate = new_rate
                        self.token_bucket = TokenBucket(self.current_rate, self.config.burst_size)
                
                elif error_rate < 0.02:  # Less than 2% errors
                    # Increase rate gradually
                    new_rate = self.current_rate * (1 + self.config.adaptive_increase_rate)
                    new_rate = min(new_rate, self.config.max_requests_per_second)
                    
                    if new_rate != self.current_rate:
                        self.logger.info(f"Increasing rate from {self.current_rate:.2f} to {new_rate:.2f}")
                        self.current_rate = new_rate
                        self.token_bucket = TokenBucket(self.current_rate, self.config.burst_size)
            
            # Reset counters
            self.success_count = 0
            self.error_count = 0
            self.last_adjustment = now
    
    async def _fixed_window_acquire(self) -> bool:
        """Fixed window rate limiting"""
        
        now = time.time()
        window_start = int(now / self.config.window_size) * self.config.window_size
        
        # Simple implementation - would need proper window tracking in production
        return True  # Simplified for demo
    
    async def report_result(self, success: bool):
        """Report request result for adaptive adjustment"""
        
        async with self._lock:
            if success:
                self.success_count += 1
            else:
                self.error_count += 1


class ErrorManager:
    """Central error management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.classifier = ErrorClassifier()
        self.retry_configs = self._load_retry_configs()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, AdaptiveRateLimiter] = {}
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.error_statistics: Dict[str, Any] = defaultdict(int)
        self.max_error_records = self.config.get('max_error_records', 1000)
        
        # Error storage
        self.error_log_file = Path(self.config.get('error_log_file', 'errors.jsonl'))
        
    def _load_retry_configs(self) -> Dict[ErrorCategory, RetryConfig]:
        """Load retry configurations for different error categories"""
        
        return {
            ErrorCategory.NETWORK: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=5,
                base_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0
            ),
            ErrorCategory.RATE_LIMIT: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=10,
                base_delay=5.0,
                max_delay=300.0,
                exponential_base=1.5
            ),
            ErrorCategory.BOT_DETECTION: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=30.0,
                max_delay=600.0,
                exponential_base=2.0
            ),
            ErrorCategory.PARSING: RetryConfig(
                strategy=RetryStrategy.NO_RETRY,
                max_attempts=1,
                base_delay=0,
                max_delay=0
            ),
            ErrorCategory.VALIDATION: RetryConfig(
                strategy=RetryStrategy.NO_RETRY,
                max_attempts=1,
                base_delay=0,
                max_delay=0
            ),
            ErrorCategory.AUTHENTICATION: RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=3,
                base_delay=10.0,
                max_delay=30.0
            )
        }
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                          operation_name: str = "unknown") -> ErrorRecord:
        """Handle error with classification and logging"""
        
        # Classify error
        category, severity = self.classifier.classify_error(error, context)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=self._generate_error_id(error, context),
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context or {},
            stack_trace=traceback.format_exc(),
            metadata={'operation': operation_name}
        )
        
        # Store error record
        await self._store_error_record(error_record)
        
        # Update statistics
        self.error_statistics[category.value] += 1
        self.error_statistics[severity.value] += 1
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error in {operation_name}: {error}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error in {operation_name}: {error}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error in {operation_name}: {error}")
        else:
            self.logger.info(f"Low severity error in {operation_name}: {error}")
        
        return error_record
    
    async def execute_with_protection(self, func: Callable, operation_name: str,
                                    context: Dict[str, Any] = None,
                                    use_circuit_breaker: bool = True,
                                    use_rate_limiter: bool = True) -> Any:
        """Execute function with full error protection"""
        
        context = context or {}
        
        # Rate limiting
        if use_rate_limiter:
            rate_limiter = self._get_rate_limiter(operation_name)
            if not await rate_limiter.acquire():
                await asyncio.sleep(0.1)  # Brief delay if rate limited
        
        # Circuit breaker protection
        if use_circuit_breaker:
            circuit_breaker = self._get_circuit_breaker(operation_name)
            
            try:
                result = await circuit_breaker.call(self._execute_with_retry, func, operation_name, context)
                
                # Report success to rate limiter
                if use_rate_limiter:
                    await rate_limiter.report_result(True)
                
                return result
                
            except Exception as e:
                # Report failure to rate limiter
                if use_rate_limiter:
                    await rate_limiter.report_result(False)
                
                # Handle the error
                await self.handle_error(e, context, operation_name)
                raise e
        else:
            return await self._execute_with_retry(func, operation_name, context)
    
    async def _execute_with_retry(self, func: Callable, operation_name: str,
                                context: Dict[str, Any]) -> Any:
        """Execute function with retry logic"""
        
        async def wrapper():
            try:
                return await func()
            except Exception as e:
                # Classify error to determine retry strategy
                category, _ = self.classifier.classify_error(e, context)
                retry_config = self.retry_configs.get(category, self.retry_configs[ErrorCategory.NETWORK])
                
                # Create retry handler and execute
                retry_handler = RetryHandler(retry_config)
                return await retry_handler.execute_with_retry(func)
        
        return await wrapper()
    
    def _get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation"""
        
        if operation_name not in self.circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=self.config.get('circuit_breaker_failure_threshold', 5),
                recovery_timeout=self.config.get('circuit_breaker_recovery_timeout', 60)
            )
            self.circuit_breakers[operation_name] = CircuitBreaker(config)
        
        return self.circuit_breakers[operation_name]
    
    def _get_rate_limiter(self, operation_name: str) -> AdaptiveRateLimiter:
        """Get or create rate limiter for operation"""
        
        if operation_name not in self.rate_limiters:
            config = RateLimitConfig(
                strategy=RateLimitStrategy.ADAPTIVE,
                requests_per_second=self.config.get('default_rate_limit', 2.0),
                burst_size=self.config.get('burst_size', 5)
            )
            self.rate_limiters[operation_name] = AdaptiveRateLimiter(config)
        
        return self.rate_limiters[operation_name]
    
    async def _store_error_record(self, error_record: ErrorRecord):
        """Store error record in memory and persistent storage"""
        
        # Add to in-memory storage
        self.error_records.append(error_record)
        
        # Trim old records
        if len(self.error_records) > self.max_error_records:
            self.error_records = self.error_records[-self.max_error_records:]
        
        # Persist to file
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(json.dumps(asdict(error_record), default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write error to log file: {e}")
    
    def _generate_error_id(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """Generate unique error ID"""
        
        error_signature = f"{type(error).__name__}:{str(error)}"
        if context:
            error_signature += f":{str(sorted(context.items()))}"
        
        return hashlib.md5(error_signature.encode()).hexdigest()[:12]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        recent_errors = [
            err for err in self.error_records
            if (datetime.now() - err.timestamp).seconds < 3600  # Last hour
        ]
        
        category_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        
        for error in recent_errors:
            category_stats[error.category.value] += 1
            severity_stats[error.severity.value] += 1
        
        return {
            'total_errors': len(self.error_records),
            'recent_errors': len(recent_errors),
            'category_distribution': dict(category_stats),
            'severity_distribution': dict(severity_stats),
            'circuit_breaker_states': {
                name: cb.state.value 
                for name, cb in self.circuit_breakers.items()
            },
            'rate_limiter_rates': {
                name: rl.current_rate
                for name, rl in self.rate_limiters.items()
            }
        }
    
    async def cleanup_old_errors(self, days_old: int = 7):
        """Clean up old error records"""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Clean in-memory records
        self.error_records = [
            err for err in self.error_records
            if err.timestamp > cutoff_date
        ]
        
        self.logger.info(f"Cleaned up error records older than {days_old} days")


# Decorator for easy error handling
def with_error_handling(operation_name: str = None, 
                       use_circuit_breaker: bool = True,
                       use_rate_limiter: bool = True):
    """Decorator for automatic error handling"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get error manager instance (would be injected in real implementation)
            error_manager = ErrorManager()  # Simplified
            
            op_name = operation_name or func.__name__
            context = {'function': func.__name__, 'args_count': len(args)}
            
            return await error_manager.execute_with_protection(
                lambda: func(*args, **kwargs),
                op_name,
                context,
                use_circuit_breaker,
                use_rate_limiter
            )
        
        return wrapper
    return decorator