"""
Advanced Anti-Detection System for eBay Jewelry Scraper
Comprehensive bot prevention with adaptive strategies and intelligent behavior simulation
"""

import asyncio
import random
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from playwright.async_api import Browser, BrowserContext, Page
import user_agents
from collections import defaultdict, deque
import logging


class DetectionLevel(Enum):
    """Detection severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProxyType(Enum):
    """Proxy types for rotation"""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"


class RequestType(Enum):
    """Request type classifications"""
    SEARCH_PAGE = "search_page"
    LISTING_PAGE = "listing_page"
    IMAGE_DOWNLOAD = "image_download"
    API_CALL = "api_call"


@dataclass
class ProxyConfig:
    """Proxy configuration with health metrics"""
    address: str
    port: int
    proxy_type: ProxyType
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    is_banned: bool = False
    ban_until: Optional[datetime] = None


@dataclass
class DetectionEvent:
    """Bot detection event record"""
    timestamp: datetime
    detection_level: DetectionLevel
    user_agent: str
    proxy: Optional[str]
    url: str
    response_code: int
    response_time: float
    indicators: List[str]
    metadata: Dict[str, Any]


@dataclass
class BrowserFingerprint:
    """Browser fingerprint configuration"""
    viewport_width: int
    viewport_height: int
    user_agent: str
    platform: str
    language: str
    timezone: str
    screen_resolution: str
    color_depth: int
    device_memory: Optional[int] = None
    hardware_concurrency: Optional[int] = None
    max_touch_points: Optional[int] = None


class UserAgentManager:
    """Intelligent user agent management with success tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # User agent pools
        self.desktop_agents = self._generate_desktop_agents()
        self.mobile_agents = self._generate_mobile_agents()
        
        # Success tracking
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'success_rate': 1.0,
            'last_used': None,
            'consecutive_failures': 0,
            'is_blocked': False,
            'block_until': None
        })
        
        # Configuration
        self.rotation_frequency = config.get('rotation_frequency', 50)
        self.max_consecutive_failures = config.get('max_consecutive_failures', 5)
        self.block_duration = config.get('block_duration', 3600)  # 1 hour
        
        # Current state
        self.request_count = 0
        self.current_agent = random.choice(self.desktop_agents)
    
    def _generate_desktop_agents(self) -> List[str]:
        """Generate realistic desktop user agents"""
        base_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{version}) Gecko/20100101 Firefox/{version}",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{version}) Gecko/20100101 Firefox/{version}",
        ]
        
        # Generate with various Chrome/Firefox versions
        agents = []
        chrome_versions = ['120.0.0.0', '119.0.0.0', '118.0.0.0', '117.0.0.0']
        firefox_versions = ['120.0', '119.0', '118.0', '117.0']
        
        for agent in base_agents:
            if 'Chrome' in agent:
                for version in chrome_versions:
                    agents.append(agent.format(version=version))
            elif 'Firefox' in agent:
                for version in firefox_versions:
                    agents.append(agent.format(version=version))
        
        return agents
    
    def _generate_mobile_agents(self) -> List[str]:
        """Generate realistic mobile user agents"""
        return [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        ]
    
    async def get_user_agent(self, mobile: bool = False) -> str:
        """Get optimal user agent with rotation and success-based selection"""
        self.request_count += 1
        
        # Check if rotation is needed
        if self.request_count % self.rotation_frequency == 0:
            await self._rotate_user_agent(mobile)
        
        return self.current_agent
    
    async def _rotate_user_agent(self, mobile: bool = False):
        """Rotate to best performing user agent"""
        agent_pool = self.mobile_agents if mobile else self.desktop_agents
        
        # Filter out blocked agents
        available_agents = []
        current_time = datetime.now()
        
        for agent in agent_pool:
            metrics = self.agent_metrics[agent]
            if metrics['is_blocked']:
                if metrics['block_until'] and current_time > metrics['block_until']:
                    # Unblock agent
                    metrics['is_blocked'] = False
                    metrics['block_until'] = None
                    metrics['consecutive_failures'] = 0
                    available_agents.append(agent)
                # Else: still blocked
            else:
                available_agents.append(agent)
        
        if not available_agents:
            # All agents blocked - emergency mode
            self.logger.warning("All user agents blocked - using random agent")
            self.current_agent = random.choice(agent_pool)
            return
        
        # Select based on success rates (weighted random selection)
        weights = []
        for agent in available_agents:
            metrics = self.agent_metrics[agent]
            # Weight based on success rate and recency
            weight = metrics['success_rate']
            if metrics['last_used']:
                time_since_used = (current_time - metrics['last_used']).seconds
                # Slight preference for less recently used agents
                weight *= (1 + min(time_since_used / 3600, 0.5))
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            self.current_agent = random.choices(available_agents, weights=normalized_weights)[0]
        else:
            self.current_agent = random.choice(available_agents)
        
        self.agent_metrics[self.current_agent]['last_used'] = current_time
    
    async def report_success(self, user_agent: str, success: bool):
        """Report user agent success/failure for adaptive selection"""
        metrics = self.agent_metrics[user_agent]
        
        if success:
            metrics['success_count'] += 1
            metrics['consecutive_failures'] = 0
        else:
            metrics['failure_count'] += 1
            metrics['consecutive_failures'] += 1
        
        # Recalculate success rate
        total_requests = metrics['success_count'] + metrics['failure_count']
        metrics['success_rate'] = metrics['success_count'] / total_requests if total_requests > 0 else 1.0
        
        # Block agent if too many consecutive failures
        if metrics['consecutive_failures'] >= self.max_consecutive_failures:
            metrics['is_blocked'] = True
            metrics['block_until'] = datetime.now() + timedelta(seconds=self.block_duration)
            self.logger.warning(f"Blocked user agent due to consecutive failures: {user_agent[:50]}...")


class ProxyRotator:
    """Advanced proxy rotation with health monitoring and geographic distribution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        self.proxies: List[ProxyConfig] = []
        self.current_index = 0
        self.rotation_frequency = config.get('rotation_frequency', 25)
        self.health_check_interval = config.get('health_check_interval', 300)
        self.max_consecutive_failures = config.get('max_consecutive_failures', 3)
        self.ban_duration = config.get('ban_duration', 1800)  # 30 minutes
        
        # Load proxies from configuration
        self._load_proxies(config.get('proxy_list', []))
        
        # Health monitoring
        self.last_health_check = datetime.now()
        self.request_count = 0
        
    def _load_proxies(self, proxy_list: List[Dict[str, Any]]):
        """Load proxy configurations"""
        for proxy_data in proxy_list:
            proxy = ProxyConfig(**proxy_data)
            self.proxies.append(proxy)
        
        if self.proxies:
            random.shuffle(self.proxies)  # Randomize initial order
            self.logger.info(f"Loaded {len(self.proxies)} proxies")
        else:
            self.logger.warning("No proxies configured - direct connections only")
    
    async def get_proxy(self) -> Optional[ProxyConfig]:
        """Get next healthy proxy with automatic rotation"""
        if not self.proxies:
            return None
        
        self.request_count += 1
        
        # Health check if needed
        if (datetime.now() - self.last_health_check).seconds > self.health_check_interval:
            await self._health_check_proxies()
        
        # Rotation logic
        if self.request_count % self.rotation_frequency == 0:
            await self._rotate_proxy()
        
        # Find healthy proxy
        attempts = 0
        max_attempts = len(self.proxies)
        
        while attempts < max_attempts:
            proxy = self.proxies[self.current_index]
            
            # Check if proxy is available
            if not proxy.is_banned and self._is_proxy_healthy(proxy):
                proxy.last_used = datetime.now()
                return proxy
            
            # Try next proxy
            self._rotate_index()
            attempts += 1
        
        # All proxies unavailable
        self.logger.error("All proxies unavailable - returning None")
        return None
    
    def _is_proxy_healthy(self, proxy: ProxyConfig) -> bool:
        """Check if proxy is considered healthy"""
        if proxy.is_banned:
            if proxy.ban_until and datetime.now() > proxy.ban_until:
                # Unban proxy
                proxy.is_banned = False
                proxy.ban_until = None
                proxy.consecutive_failures = 0
                return True
            return False
        
        # Consider proxy healthy if success rate is acceptable
        return proxy.success_rate > 0.5 and proxy.consecutive_failures < self.max_consecutive_failures
    
    async def _rotate_proxy(self):
        """Rotate to next proxy"""
        self._rotate_index()
    
    def _rotate_index(self):
        """Move to next proxy index"""
        self.current_index = (self.current_index + 1) % len(self.proxies)
    
    async def _health_check_proxies(self):
        """Perform health checks on all proxies"""
        self.logger.info("Performing proxy health checks...")
        
        for proxy in self.proxies:
            if proxy.is_banned:
                continue
                
            try:
                # Simple health check - can be enhanced with actual test requests
                start_time = time.time()
                
                # Test connection (simplified)
                test_url = "http://httpbin.org/ip"
                proxy_url = f"{proxy.proxy_type.value}://"
                if proxy.username and proxy.password:
                    proxy_url += f"{proxy.username}:{proxy.password}@"
                proxy_url += f"{proxy.address}:{proxy.port}"
                
                response = requests.get(
                    test_url,
                    proxies={proxy.proxy_type.value: proxy_url},
                    timeout=10
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    proxy.avg_response_time = (proxy.avg_response_time + response_time) / 2
                    proxy.consecutive_failures = 0
                else:
                    await self._handle_proxy_failure(proxy, "Health check failed")
                    
            except Exception as e:
                await self._handle_proxy_failure(proxy, str(e))
        
        self.last_health_check = datetime.now()
    
    async def report_proxy_result(self, proxy: ProxyConfig, success: bool, response_time: float = 0.0):
        """Report proxy performance for adaptive management"""
        proxy.total_requests += 1
        
        if success:
            proxy.consecutive_failures = 0
            if response_time > 0:
                proxy.avg_response_time = (proxy.avg_response_time + response_time) / 2
            
            # Update success rate with exponential moving average
            proxy.success_rate = 0.9 * proxy.success_rate + 0.1 * 1.0
        else:
            proxy.consecutive_failures += 1
            proxy.success_rate = 0.9 * proxy.success_rate + 0.1 * 0.0
            
            await self._handle_proxy_failure(proxy, "Request failed")
    
    async def _handle_proxy_failure(self, proxy: ProxyConfig, reason: str):
        """Handle proxy failure with potential banning"""
        self.logger.warning(f"Proxy failure: {proxy.address}:{proxy.port} - {reason}")
        
        if proxy.consecutive_failures >= self.max_consecutive_failures:
            proxy.is_banned = True
            proxy.ban_until = datetime.now() + timedelta(seconds=self.ban_duration)
            self.logger.warning(f"Banned proxy: {proxy.address}:{proxy.port} for {self.ban_duration}s")


class BrowserFingerprintRandomizer:
    """Randomize browser fingerprints to avoid detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.randomize_viewport = config.get('randomize_viewport', True)
        self.randomize_timezone = config.get('randomize_timezone', True)
        self.randomize_language = config.get('randomize_language', True)
        
        # Realistic value ranges
        self.viewport_ranges = {
            'desktop': [(1366, 768), (1920, 1080), (1536, 864), (1440, 900), (1280, 720)],
            'mobile': [(375, 667), (414, 896), (360, 640), (393, 851)]
        }
        
        self.timezones = [
            'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
            'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Asia/Tokyo', 'Asia/Shanghai'
        ]
        
        self.languages = [
            'en-US,en;q=0.9',
            'en-GB,en;q=0.9',
            'en-CA,en;q=0.9',
            'fr-FR,fr;q=0.9,en;q=0.8',
            'de-DE,de;q=0.9,en;q=0.8',
            'es-ES,es;q=0.9,en;q=0.8'
        ]
    
    async def generate_fingerprint(self, user_agent: str, mobile: bool = False) -> BrowserFingerprint:
        """Generate realistic browser fingerprint"""
        
        # Determine viewport
        if self.randomize_viewport:
            viewport_pool = self.viewport_ranges['mobile' if mobile else 'desktop']
            viewport_width, viewport_height = random.choice(viewport_pool)
            # Add small random variation
            viewport_width += random.randint(-50, 50)
            viewport_height += random.randint(-50, 50)
        else:
            viewport_width, viewport_height = (1920, 1080) if not mobile else (375, 667)
        
        # Platform from user agent
        if 'Windows' in user_agent:
            platform = 'Win32'
        elif 'Macintosh' in user_agent:
            platform = 'MacIntel'
        elif 'Linux' in user_agent:
            platform = 'Linux x86_64'
        else:
            platform = 'Win32'  # Default fallback
        
        # Language
        language = random.choice(self.languages) if self.randomize_language else 'en-US,en;q=0.9'
        
        # Timezone
        timezone = random.choice(self.timezones) if self.randomize_timezone else 'America/New_York'
        
        # Screen resolution (usually larger than viewport)
        screen_width = viewport_width + random.randint(0, 200)
        screen_height = viewport_height + random.randint(0, 200)
        
        # Hardware specs
        hardware_concurrency = random.choice([2, 4, 8, 12, 16]) if not mobile else random.choice([4, 6, 8])
        device_memory = random.choice([2, 4, 8, 16]) if not mobile else random.choice([2, 3, 4, 6])
        
        return BrowserFingerprint(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            user_agent=user_agent,
            platform=platform,
            language=language,
            timezone=timezone,
            screen_resolution=f"{screen_width}x{screen_height}",
            color_depth=24,
            device_memory=device_memory,
            hardware_concurrency=hardware_concurrency,
            max_touch_points=0 if not mobile else random.choice([5, 10])
        )
    
    async def apply_fingerprint(self, page: Page, fingerprint: BrowserFingerprint):
        """Apply fingerprint to browser page"""
        try:
            # Set viewport
            await page.set_viewport_size(fingerprint.viewport_width, fingerprint.viewport_height)
            
            # Override navigator properties
            await page.add_init_script(f"""
                Object.defineProperty(navigator, 'platform', {{
                    get: () => '{fingerprint.platform}'
                }});
                
                Object.defineProperty(navigator, 'language', {{
                    get: () => '{fingerprint.language.split(',')[0]}'
                }});
                
                Object.defineProperty(navigator, 'languages', {{
                    get: () => {json.dumps(fingerprint.language.split(','))}
                }});
                
                Object.defineProperty(navigator, 'hardwareConcurrency', {{
                    get: () => {fingerprint.hardware_concurrency}
                }});
                
                Object.defineProperty(navigator, 'deviceMemory', {{
                    get: () => {fingerprint.device_memory}
                }});
                
                Object.defineProperty(screen, 'width', {{
                    get: () => {fingerprint.screen_resolution.split('x')[0]}
                }});
                
                Object.defineProperty(screen, 'height', {{
                    get: () => {fingerprint.screen_resolution.split('x')[1]}
                }});
                
                Object.defineProperty(screen, 'colorDepth', {{
                    get: () => {fingerprint.color_depth}
                }});
                
                // Override timezone
                Date.prototype.getTimezoneOffset = function() {{
                    return -new Date().getTimezoneOffset();
                }};
            """)
            
            # Set user agent
            await page.set_extra_http_headers({
                'User-Agent': fingerprint.user_agent,
                'Accept-Language': fingerprint.language
            })
            
        except Exception as e:
            self.logger.error(f"Failed to apply browser fingerprint: {e}")


class RequestPatternManager:
    """Manage request patterns to simulate human behavior"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # Timing configuration
        self.min_delay = config.get('min_delay', 1.0)
        self.max_delay = config.get('max_delay', 5.0)
        self.burst_protection = config.get('burst_protection', True)
        self.human_simulation = config.get('human_simulation', True)
        
        # Request history
        self.request_history: deque = deque(maxlen=100)
        self.last_request_time = 0.0
        
        # Pattern analysis
        self.request_intervals: List[float] = []
        
    async def calculate_delay(self, request_type: RequestType) -> float:
        """Calculate intelligent delay based on request type and patterns"""
        current_time = time.time()
        
        # Base delay calculation
        if request_type == RequestType.SEARCH_PAGE:
            base_delay = random.uniform(2.0, 5.0)
        elif request_type == RequestType.LISTING_PAGE:
            base_delay = random.uniform(1.5, 4.0)
        elif request_type == RequestType.IMAGE_DOWNLOAD:
            base_delay = random.uniform(0.5, 2.0)
        else:
            base_delay = random.uniform(self.min_delay, self.max_delay)
        
        # Human-like variability
        if self.human_simulation:
            # Add occasional longer pauses (simulating user reading/thinking)
            if random.random() < 0.1:  # 10% chance
                base_delay += random.uniform(5.0, 15.0)
            
            # Add micro-variations
            base_delay += random.gauss(0, 0.3)
        
        # Burst protection
        if self.burst_protection:
            recent_requests = [
                req for req in self.request_history 
                if current_time - req['timestamp'] < 60  # Last minute
            ]
            
            if len(recent_requests) > 10:  # Too many recent requests
                base_delay *= 1.5  # Increase delay
        
        # Ensure minimum time since last request
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            base_delay = max(base_delay, self.min_delay - time_since_last)
        
        return max(0.1, base_delay)  # Minimum 0.1 second delay
    
    async def record_request(self, request_type: RequestType, delay: float):
        """Record request for pattern analysis"""
        current_time = time.time()
        
        self.request_history.append({
            'timestamp': current_time,
            'type': request_type,
            'delay': delay
        })
        
        if self.last_request_time > 0:
            interval = current_time - self.last_request_time
            self.request_intervals.append(interval)
            
            # Keep only recent intervals
            if len(self.request_intervals) > 50:
                self.request_intervals.pop(0)
        
        self.last_request_time = current_time
    
    async def simulate_human_interactions(self, page: Page):
        """Simulate human-like interactions on the page"""
        if not self.human_simulation:
            return
        
        try:
            # Random scroll simulation
            if random.random() < 0.7:  # 70% chance to scroll
                scroll_distance = random.randint(200, 800)
                await page.evaluate(f"window.scrollBy(0, {scroll_distance})")
                await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Random mouse movement (can't actually move cursor, but can trigger hover events)
            if random.random() < 0.3:  # 30% chance
                # Find random element to hover
                elements = await page.query_selector_all('div, img, a')
                if elements:
                    random_element = random.choice(elements)
                    await random_element.hover()
                    await asyncio.sleep(random.uniform(0.2, 0.8))
            
            # Occasional pause (simulating reading)
            if random.random() < 0.2:  # 20% chance
                await asyncio.sleep(random.uniform(2.0, 5.0))
                
        except Exception as e:
            self.logger.debug(f"Error in human interaction simulation: {e}")


class DetectionAnalyzer:
    """Analyze responses for bot detection indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # Detection patterns
        self.captcha_indicators = [
            'captcha', 'recaptcha', 'challenge', 'verify', 'robot',
            'security check', 'unusual activity'
        ]
        
        self.rate_limit_indicators = [
            'rate limit', 'too many requests', 'slow down',
            'temporarily blocked', '429', 'throttled'
        ]
        
        self.bot_detection_indicators = [
            'automated requests', 'bot detected', 'suspicious activity',
            'access denied', 'forbidden', 'blocked'
        ]
        
        # Thresholds
        self.response_time_threshold = config.get('response_time_threshold', 10.0)
        self.error_rate_threshold = config.get('error_rate_threshold', 0.3)
        
    async def analyze_response(self, response, html_content: str, 
                             response_time: float) -> DetectionEvent:
        """Analyze response for bot detection indicators"""
        
        detection_level = DetectionLevel.NONE
        indicators = []
        
        # Status code analysis
        if response.status == 403:
            detection_level = DetectionLevel.HIGH
            indicators.append("HTTP 403 Forbidden")
        elif response.status == 429:
            detection_level = DetectionLevel.MEDIUM
            indicators.append("HTTP 429 Rate Limited")
        elif response.status in [503, 502, 504]:
            detection_level = DetectionLevel.MEDIUM
            indicators.append(f"HTTP {response.status} Server Error")
        
        # Content analysis
        html_lower = html_content.lower()
        
        # Check for CAPTCHA
        for indicator in self.captcha_indicators:
            if indicator in html_lower:
                detection_level = max(detection_level, DetectionLevel.HIGH)
                indicators.append(f"CAPTCHA indicator: {indicator}")
                break
        
        # Check for rate limiting
        for indicator in self.rate_limit_indicators:
            if indicator in html_lower:
                detection_level = max(detection_level, DetectionLevel.MEDIUM)
                indicators.append(f"Rate limit indicator: {indicator}")
                break
        
        # Check for bot detection
        for indicator in self.bot_detection_indicators:
            if indicator in html_lower:
                detection_level = max(detection_level, DetectionLevel.HIGH)
                indicators.append(f"Bot detection indicator: {indicator}")
                break
        
        # Response time analysis
        if response_time > self.response_time_threshold:
            detection_level = max(detection_level, DetectionLevel.LOW)
            indicators.append(f"High response time: {response_time:.2f}s")
        
        # Content length analysis
        if len(html_content) < 1000:  # Suspiciously short content
            detection_level = max(detection_level, DetectionLevel.LOW)
            indicators.append("Suspiciously short content")
        
        return DetectionEvent(
            timestamp=datetime.now(),
            detection_level=detection_level,
            user_agent=response.headers.get('user-agent', ''),
            proxy=None,  # Would be filled by calling code
            url=str(response.url),
            response_code=response.status,
            response_time=response_time,
            indicators=indicators,
            metadata={
                'content_length': len(html_content),
                'headers': dict(response.headers)
            }
        )


class AntiDetectionSystem:
    """Main anti-detection system coordinating all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.user_agent_manager = UserAgentManager(config.get('user_agents', {}))
        self.proxy_rotator = ProxyRotator(config.get('proxies', {}))
        self.fingerprint_randomizer = BrowserFingerprintRandomizer(config.get('fingerprinting', {}))
        self.request_pattern_manager = RequestPatternManager(config.get('request_patterns', {}))
        self.detection_analyzer = DetectionAnalyzer(config.get('detection_analysis', {}))
        
        # State tracking
        self.detection_events: List[DetectionEvent] = []
        self.current_strategy = "default"
        self.strategy_change_count = 0
        
        # Adaptive thresholds
        self.adaptation_threshold = config.get('adaptation_threshold', 5)
        self.cooldown_duration = config.get('cooldown_duration', 1800)  # 30 minutes
        
    async def prepare_request(self, request_type: RequestType, 
                            mobile: bool = False) -> Dict[str, Any]:
        """Prepare request with all anti-detection measures"""
        
        # Get user agent
        user_agent = await self.user_agent_manager.get_user_agent(mobile)
        
        # Get proxy
        proxy = await self.proxy_rotator.get_proxy()
        
        # Generate fingerprint
        fingerprint = await self.fingerprint_randomizer.generate_fingerprint(user_agent, mobile)
        
        # Calculate delay
        delay = await self.request_pattern_manager.calculate_delay(request_type)
        
        return {
            'user_agent': user_agent,
            'proxy': proxy,
            'fingerprint': fingerprint,
            'delay': delay,
            'request_type': request_type
        }
    
    async def execute_request(self, page: Page, url: str, request_config: Dict[str, Any]):
        """Execute request with anti-detection measures applied"""
        
        # Apply delay
        if request_config['delay'] > 0:
            await asyncio.sleep(request_config['delay'])
        
        # Apply fingerprint
        await self.fingerprint_randomizer.apply_fingerprint(page, request_config['fingerprint'])
        
        # Navigate to URL
        start_time = time.time()
        response = await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        response_time = time.time() - start_time
        
        # Get page content
        html_content = await page.content()
        
        # Analyze response for detection
        detection_event = await self.detection_analyzer.analyze_response(
            response, html_content, response_time
        )
        detection_event.proxy = request_config['proxy'].address if request_config['proxy'] else None
        
        # Record request
        await self.request_pattern_manager.record_request(
            request_config['request_type'], request_config['delay']
        )
        
        # Handle detection if necessary
        await self._handle_detection_event(detection_event, request_config)
        
        # Simulate human interactions
        await self.request_pattern_manager.simulate_human_interactions(page)
        
        return response, html_content, detection_event
    
    async def _handle_detection_event(self, event: DetectionEvent, request_config: Dict[str, Any]):
        """Handle detection event and adapt strategy"""
        
        self.detection_events.append(event)
        
        # Report results to components
        success = event.detection_level in [DetectionLevel.NONE, DetectionLevel.LOW]
        
        await self.user_agent_manager.report_success(event.user_agent, success)
        
        if request_config['proxy']:
            await self.proxy_rotator.report_proxy_result(
                request_config['proxy'], success, event.response_time
            )
        
        # Check if adaptation is needed
        recent_detections = [
            e for e in self.detection_events[-self.adaptation_threshold:]
            if e.detection_level in [DetectionLevel.MEDIUM, DetectionLevel.HIGH, DetectionLevel.CRITICAL]
        ]
        
        if len(recent_detections) >= self.adaptation_threshold * 0.6:  # 60% detection rate
            await self._adapt_strategy()
    
    async def _adapt_strategy(self):
        """Adapt anti-detection strategy based on recent events"""
        
        self.strategy_change_count += 1
        self.logger.warning(f"Adapting strategy (change #{self.strategy_change_count})")
        
        # Implement emergency cooldown
        if self.strategy_change_count % 3 == 0:  # Every 3rd adaptation
            cooldown_time = min(self.cooldown_duration * (self.strategy_change_count // 3), 3600)
            self.logger.warning(f"Emergency cooldown for {cooldown_time} seconds")
            await asyncio.sleep(cooldown_time)
        
        # Reset component states for fresh start
        self.user_agent_manager.request_count = 0
        self.proxy_rotator.request_count = 0
        
        # Clear recent detection events
        self.detection_events = self.detection_events[-10:]  # Keep only last 10
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anti-detection system statistics"""
        
        recent_events = self.detection_events[-50:]  # Last 50 events
        
        detection_counts = defaultdict(int)
        for event in recent_events:
            detection_counts[event.detection_level.value] += 1
        
        return {
            'total_requests': len(self.detection_events),
            'detection_distribution': dict(detection_counts),
            'strategy_changes': self.strategy_change_count,
            'current_strategy': self.current_strategy,
            'user_agent_stats': len(self.user_agent_manager.agent_metrics),
            'proxy_stats': {
                'total_proxies': len(self.proxy_rotator.proxies),
                'healthy_proxies': len([p for p in self.proxy_rotator.proxies if not p.is_banned]),
                'banned_proxies': len([p for p in self.proxy_rotator.proxies if p.is_banned])
            }
        }