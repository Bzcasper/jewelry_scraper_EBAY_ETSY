"""
Advanced Rate Limiting and Retry Logic for eBay Jewelry Scraping

Comprehensive rate limiting system with intelligent backoff strategies,
concurrent request management, and adaptive retry mechanisms.

Features:
- Adaptive rate limiting based on response patterns
- Exponential backoff with jitter
- Circuit breaker pattern
- Concurrent request semaphore control
- Request queue management
- Retry strategies with different policies
- Anti-detection awareness
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import deque, defaultdict
import uuid


class RetryPolicy(Enum):
    """Retry policy strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    ADAPTIVE = "adaptive"
    CIRCUIT_BREAKER = "circuit_breaker"


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_RATE = "fixed_rate"
    ADAPTIVE_RATE = "adaptive_rate"
    BURST_CONTROL = "burst_control"
    TIME_WINDOW = "time_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RequestMetrics:
    """Metrics for request tracking"""
    timestamp: datetime
    response_time: float
    status_code: Optional[int] = None
    success: bool = True
    error_type: Optional[str] = None
    detection_risk: float = 0.0
    retry_count: int = 0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    # Basic rate limits
    requests_per_second: float = 1.0
    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    
    # Burst control
    max_burst_size: int = 5
    burst_recovery_time: float = 60.0
    
    # Adaptive settings
    min_delay: float = 1.0
    max_delay: float = 300.0
    adaptive_threshold: float = 0.7
    
    # Concurrent limits
    max_concurrent_requests: int = 3
    max_queue_size: int = 100
    
    # Detection avoidance
    detection_cooldown: float = 600.0  # 10 minutes
    risk_threshold: float = 0.8
    
    # Randomization
    jitter_factor: float = 0.1
    randomize_delays: bool = True


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    
    # Retry conditions
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 503, 502, 504])
    retry_on_timeouts: bool = True
    retry_on_connection_errors: bool = True
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_time: float = 300.0


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for request failure handling"""
    failure_threshold: int
    recovery_timeout: float
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def should_attempt_request(self) -> bool:
        """Check if request should be attempted"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery time has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill bucket based on elapsed time"""
        current_time = time.time()
        elapsed = current_time - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time
    
    def available_tokens(self) -> int:
        """Get number of available tokens"""
        self._refill()
        return int(self.tokens)


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple strategies and adaptive behavior
    """
    
    def __init__(self, 
                 rate_config: Optional[RateLimitConfig] = None,
                 retry_config: Optional[RetryConfig] = None):
        """
        Initialize rate limiter
        
        Args:
            rate_config: Rate limiting configuration
            retry_config: Retry configuration
        """
        self.rate_config = rate_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()
        
        # Request tracking
        self.request_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=100)
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=self.rate_config.max_queue_size)
        
        # Concurrent request control
        self.semaphore = asyncio.Semaphore(self.rate_config.max_concurrent_requests)
        
        # Token bucket for burst control
        self.token_bucket = TokenBucket(
            capacity=self.rate_config.max_burst_size,
            refill_rate=self.rate_config.requests_per_second
        )
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.retry_config.circuit_breaker_threshold,
            recovery_timeout=self.retry_config.circuit_breaker_recovery_time
        )
        
        # Adaptive rate tracking
        self.current_delay = self.rate_config.min_delay
        self.last_request_time = 0.0
        self.detection_risk_score = 0.0
        self.cooldown_until: Optional[float] = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retries_performed': 0,
            'rate_limited_requests': 0,
            'circuit_breaker_trips': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_limits(self,
                                 request_func: Callable[..., Awaitable[Any]],
                                 *args,
                                 **kwargs) -> Any:
        """
        Execute request with rate limiting and retry logic
        
        Args:
            request_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        request_id = str(uuid.uuid4())[:8]
        
        # Check circuit breaker
        if not self.circuit_breaker.should_attempt_request():
            self.stats['circuit_breaker_trips'] += 1
            raise Exception("Circuit breaker is open - requests blocked")
        
        # Check cooldown period
        if self.cooldown_until and time.time() < self.cooldown_until:
            cooldown_remaining = self.cooldown_until - time.time()
            self.logger.warning(f"In detection cooldown, waiting {cooldown_remaining:.1f}s")
            await asyncio.sleep(cooldown_remaining)
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.retry_config.max_retries:
            try:
                # Apply rate limiting
                await self._apply_rate_limits(request_id)
                
                # Execute request with semaphore control
                async with self.semaphore:
                    start_time = time.time()
                    
                    result = await request_func(*args, **kwargs)
                    
                    # Record successful request
                    response_time = time.time() - start_time
                    await self._record_request_success(response_time, retry_count)
                    
                    self.circuit_breaker.record_success()
                    self.stats['successful_requests'] += 1
                    
                    return result
                    
            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time if 'start_time' in locals() else 0
                
                # Record failed request
                await self._record_request_failure(e, response_time, retry_count)
                
                # Check if we should retry
                if not self._should_retry(e, retry_count):
                    break
                
                # Calculate retry delay
                retry_delay = self._calculate_retry_delay(retry_count, e)
                
                self.logger.warning(
                    f"Request {request_id} failed (attempt {retry_count + 1}), "
                    f"retrying in {retry_delay:.1f}s: {str(e)[:100]}"
                )
                
                await asyncio.sleep(retry_delay)
                retry_count += 1
                self.stats['retries_performed'] += 1
        
        # All retries exhausted
        self.circuit_breaker.record_failure()
        self.stats['failed_requests'] += 1
        
        if last_exception:
            raise last_exception
        else:
            raise Exception("Request failed after all retry attempts")
    
    async def _apply_rate_limits(self, request_id: str):
        """Apply rate limiting before request"""
        
        # Token bucket check
        if not await self.token_bucket.consume():
            self.stats['rate_limited_requests'] += 1
            wait_time = 1.0 / self.rate_config.requests_per_second
            self.logger.debug(f"Token bucket depleted, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # Time-based rate limiting
        current_time = time.time()
        
        # Check requests per second
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.rate_config.requests_per_second
            
            if time_since_last < min_interval:
                delay = min_interval - time_since_last
                
                # Apply adaptive delay adjustment
                delay = self._apply_adaptive_delay(delay)
                
                # Add jitter if enabled
                if self.rate_config.randomize_delays:
                    jitter = delay * self.rate_config.jitter_factor * random.random()
                    delay += jitter
                
                self.logger.debug(f"Rate limiting delay: {delay:.2f}s")
                await asyncio.sleep(delay)
        
        # Check hourly and minute limits
        await self._check_time_window_limits()
        
        self.last_request_time = time.time()
        self.stats['total_requests'] += 1
    
    def _apply_adaptive_delay(self, base_delay: float) -> float:
        """Apply adaptive delay based on recent performance"""
        
        # Increase delay if detection risk is high
        if self.detection_risk_score > self.rate_config.risk_threshold:
            multiplier = 1 + (self.detection_risk_score - self.rate_config.risk_threshold) * 5
            base_delay *= multiplier
            self.logger.debug(f"High detection risk ({self.detection_risk_score:.2f}), "
                            f"increasing delay by {multiplier:.1f}x")
        
        # Check recent error rate
        recent_errors = self._get_recent_error_rate()
        if recent_errors > 0.3:  # More than 30% errors in recent requests
            base_delay *= (1 + recent_errors)
            self.logger.debug(f"High error rate ({recent_errors:.1%}), increasing delay")
        
        # Ensure delay is within bounds
        return max(
            self.rate_config.min_delay,
            min(self.rate_config.max_delay, base_delay)
        )
    
    async def _check_time_window_limits(self):
        """Check minute and hour request limits"""
        current_time = time.time()
        
        # Clean old requests from history
        cutoff_hour = current_time - 3600
        cutoff_minute = current_time - 60
        
        # Count recent requests
        minute_requests = sum(1 for req in self.request_history 
                             if req.timestamp.timestamp() > cutoff_minute)
        hour_requests = sum(1 for req in self.request_history 
                           if req.timestamp.timestamp() > cutoff_hour)
        
        # Check limits
        if minute_requests >= self.rate_config.requests_per_minute:
            wait_time = 60 - (current_time % 60)
            self.logger.warning(f"Minute limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        if hour_requests >= self.rate_config.requests_per_hour:
            wait_time = 3600 - (current_time % 3600)
            self.logger.warning(f"Hour limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
    
    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if request should be retried"""
        
        if retry_count >= self.retry_config.max_retries:
            return False
        
        error_str = str(error).lower()
        
        # Check for specific retry conditions
        if hasattr(error, 'status_code'):
            if error.status_code in self.retry_config.retry_on_status_codes:
                return True
        
        # Check for timeout errors
        if self.retry_config.retry_on_timeouts and 'timeout' in error_str:
            return True
        
        # Check for connection errors
        if (self.retry_config.retry_on_connection_errors and 
            any(term in error_str for term in ['connection', 'network', 'unreachable'])):
            return True
        
        # Rate limiting errors
        if any(term in error_str for term in ['rate limit', 'too many requests', '429']):
            return True
        
        # eBay-specific errors that should be retried
        if any(term in error_str for term in ['blocked', 'captcha', 'suspicious activity']):
            # These are detection-related errors - trigger cooldown
            self._trigger_detection_cooldown()
            return True
        
        return False
    
    def _calculate_retry_delay(self, retry_count: int, error: Exception) -> float:
        """Calculate delay before retry"""
        
        if self.retry_config.max_retries == 0:
            return 0
        
        error_str = str(error).lower()
        
        # Special handling for rate limiting
        if 'rate limit' in error_str or '429' in error_str:
            # Extract retry-after header if available
            retry_after = self._extract_retry_after(error)
            if retry_after:
                return min(retry_after, self.retry_config.max_delay)
        
        # Detection-related errors need longer delays
        if any(term in error_str for term in ['blocked', 'captcha', 'suspicious']):
            base_delay = self.retry_config.base_delay * 10  # Much longer delay
        else:
            base_delay = self.retry_config.base_delay
        
        # Apply retry strategy
        if self.retry_config.max_retries == 1:
            # Fixed delay for single retry
            delay = base_delay
        else:
            # Exponential backoff
            delay = base_delay * (self.retry_config.exponential_base ** retry_count)
        
        # Add jitter
        jitter = delay * self.retry_config.jitter_factor * random.random()
        delay += jitter
        
        # Ensure within bounds
        return min(delay, self.retry_config.max_delay)
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after value from error"""
        
        # Try to extract from HTTP headers if available
        if hasattr(error, 'response') and hasattr(error.response, 'headers'):
            retry_after = error.response.headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        
        # Try to extract from error message
        error_str = str(error)
        retry_match = re.search(r'retry.{0,10}(\d+)', error_str, re.IGNORECASE)
        if retry_match:
            try:
                return float(retry_match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _trigger_detection_cooldown(self):
        """Trigger cooldown period due to detection risk"""
        
        self.detection_risk_score = 1.0
        self.cooldown_until = time.time() + self.rate_config.detection_cooldown
        
        self.logger.warning(
            f"Detection risk triggered, entering cooldown for "
            f"{self.rate_config.detection_cooldown}s"
        )
    
    async def _record_request_success(self, response_time: float, retry_count: int):
        """Record successful request metrics"""
        
        metrics = RequestMetrics(
            timestamp=datetime.now(),
            response_time=response_time,
            success=True,
            retry_count=retry_count
        )
        
        self.request_history.append(metrics)
        
        # Gradually decrease detection risk on successful requests
        self.detection_risk_score *= 0.95
        self.detection_risk_score = max(0.0, self.detection_risk_score)
    
    async def _record_request_failure(self, 
                                    error: Exception, 
                                    response_time: float, 
                                    retry_count: int):
        """Record failed request metrics"""
        
        error_type = type(error).__name__
        detection_risk = 0.0
        
        # Assess detection risk based on error type
        error_str = str(error).lower()
        if any(term in error_str for term in ['blocked', 'captcha', 'suspicious', 'robot']):
            detection_risk = 0.8
        elif 'rate limit' in error_str or '429' in error_str:
            detection_risk = 0.4
        elif hasattr(error, 'status_code') and error.status_code in [403, 429]:
            detection_risk = 0.3
        
        metrics = RequestMetrics(
            timestamp=datetime.now(),
            response_time=response_time,
            success=False,
            error_type=error_type,
            detection_risk=detection_risk,
            retry_count=retry_count
        )
        
        self.request_history.append(metrics)
        self.error_history.append(metrics)
        
        # Update detection risk score
        self.detection_risk_score = min(1.0, self.detection_risk_score + detection_risk)
    
    def _get_recent_error_rate(self, window_minutes: int = 10) -> float:
        """Calculate error rate in recent time window"""
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_requests = [req for req in self.request_history 
                          if req.timestamp > cutoff_time]
        
        if not recent_requests:
            return 0.0
        
        error_count = sum(1 for req in recent_requests if not req.success)
        return error_count / len(recent_requests)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status"""
        
        current_time = time.time()
        
        return {
            'current_delay': self.current_delay,
            'detection_risk_score': self.detection_risk_score,
            'in_cooldown': bool(self.cooldown_until and current_time < self.cooldown_until),
            'cooldown_remaining': max(0, self.cooldown_until - current_time) if self.cooldown_until else 0,
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'available_tokens': self.token_bucket.available_tokens(),
            'concurrent_requests': self.rate_config.max_concurrent_requests - self.semaphore._value,
            'recent_error_rate': self._get_recent_error_rate(),
            'stats': self.stats.copy()
        }
    
    def adjust_rate_limits(self, new_config: RateLimitConfig):
        """Dynamically adjust rate limiting configuration"""
        
        old_concurrent = self.rate_config.max_concurrent_requests
        self.rate_config = new_config
        
        # Adjust semaphore if concurrent limit changed
        if new_config.max_concurrent_requests != old_concurrent:
            self.semaphore = asyncio.Semaphore(new_config.max_concurrent_requests)
        
        # Update token bucket
        self.token_bucket = TokenBucket(
            capacity=new_config.max_burst_size,
            refill_rate=new_config.requests_per_second
        )
        
        self.logger.info(f"Rate limits adjusted: {new_config}")
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {key: 0 for key in self.stats}
        self.detection_risk_score = 0.0
        self.cooldown_until = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.retry_config.circuit_breaker_threshold,
            recovery_timeout=self.retry_config.circuit_breaker_recovery_time
        )
        
        self.logger.info("Rate limiter stats reset")


# Convenience function for creating pre-configured rate limiters
def create_ebay_rate_limiter(aggressiveness: str = "standard") -> AdvancedRateLimiter:
    """
    Create rate limiter configured for eBay scraping
    
    Args:
        aggressiveness: "conservative", "standard", or "aggressive"
        
    Returns:
        Configured AdvancedRateLimiter
    """
    
    configs = {
        "conservative": {
            "rate": RateLimitConfig(
                requests_per_second=0.5,
                requests_per_minute=20,
                requests_per_hour=500,
                max_concurrent_requests=1,
                min_delay=2.0,
                max_delay=600.0,
                detection_cooldown=1200.0
            ),
            "retry": RetryConfig(
                max_retries=5,
                base_delay=2.0,
                max_delay=120.0,
                circuit_breaker_recovery_time=600.0
            )
        },
        "standard": {
            "rate": RateLimitConfig(
                requests_per_second=1.0,
                requests_per_minute=30,
                requests_per_hour=1000,
                max_concurrent_requests=2,
                min_delay=1.0,
                max_delay=300.0,
                detection_cooldown=600.0
            ),
            "retry": RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=60.0,
                circuit_breaker_recovery_time=300.0
            )
        },
        "aggressive": {
            "rate": RateLimitConfig(
                requests_per_second=2.0,
                requests_per_minute=60,
                requests_per_hour=2000,
                max_concurrent_requests=3,
                min_delay=0.5,
                max_delay=120.0,
                detection_cooldown=300.0
            ),
            "retry": RetryConfig(
                max_retries=2,
                base_delay=0.5,
                max_delay=30.0,
                circuit_breaker_recovery_time=120.0
            )
        }
    }
    
    config = configs.get(aggressiveness, configs["standard"])
    
    return AdvancedRateLimiter(
        rate_config=config["rate"],
        retry_config=config["retry"]
    )