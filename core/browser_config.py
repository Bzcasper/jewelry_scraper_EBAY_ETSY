"""
Advanced Browser Configuration for eBay Jewelry Scraping

Comprehensive browser configuration system with anti-detection measures,
fingerprint randomization, and stealth browsing capabilities.

Features:
- Multiple anti-detection levels
- Browser fingerprint randomization
- User agent rotation
- Proxy configuration
- Performance optimization
- Mobile/desktop simulation
"""

import random
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

from crawl4ai import BrowserConfig


class BrowserType(Enum):
    """Supported browser types"""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class DeviceProfile(Enum):
    """Device simulation profiles"""
    DESKTOP_WINDOWS = "desktop_windows"
    DESKTOP_MAC = "desktop_mac"
    DESKTOP_LINUX = "desktop_linux"
    MOBILE_ANDROID = "mobile_android"
    MOBILE_IOS = "mobile_ios"
    TABLET_ANDROID = "tablet_android"
    TABLET_IOS = "tablet_ios"


@dataclass
class ViewportConfig:
    """Viewport configuration"""
    width: int
    height: int
    device_scale_factor: float = 1.0
    is_mobile: bool = False
    has_touch: bool = False


@dataclass
class ProxyConfig:
    """Proxy configuration"""
    server: str
    username: Optional[str] = None
    password: Optional[str] = None
    bypass: Optional[List[str]] = None


@dataclass
class BrowserFingerprint:
    """Browser fingerprint configuration"""
    user_agent: str
    viewport: ViewportConfig
    timezone: str
    language: str
    platform: str
    screen_resolution: tuple
    color_depth: int
    pixel_ratio: float
    hardware_concurrency: int
    memory: int
    webgl_vendor: str
    webgl_renderer: str


class AdvancedBrowserConfigurator:
    """
    Advanced browser configuration manager with anti-detection capabilities
    """
    
    # Comprehensive user agent lists
    USER_AGENTS = {
        'windows_chrome': [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ],
        'mac_chrome': [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ],
        'mac_safari': [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15'
        ],
        'windows_firefox': [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0'
        ],
        'mobile_android': [
            'Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
            'Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36'
        ],
        'mobile_ios': [
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'
        ]
    }
    
    # Viewport configurations by device
    VIEWPORTS = {
        DeviceProfile.DESKTOP_WINDOWS: ViewportConfig(1920, 1080, 1.0, False, False),
        DeviceProfile.DESKTOP_MAC: ViewportConfig(2560, 1440, 2.0, False, False),
        DeviceProfile.DESKTOP_LINUX: ViewportConfig(1920, 1080, 1.0, False, False),
        DeviceProfile.MOBILE_ANDROID: ViewportConfig(393, 851, 2.75, True, True),
        DeviceProfile.MOBILE_IOS: ViewportConfig(414, 896, 3.0, True, True),
        DeviceProfile.TABLET_ANDROID: ViewportConfig(768, 1024, 2.0, True, True),
        DeviceProfile.TABLET_IOS: ViewportConfig(820, 1180, 2.0, True, True)
    }
    
    # Timezone options
    TIMEZONES = [
        'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
        'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Rome',
        'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Seoul', 'Australia/Sydney'
    ]
    
    # Language preferences
    LANGUAGES = [
        'en-US,en;q=0.9',
        'en-GB,en;q=0.9',
        'en-US,en;q=0.8,es;q=0.6',
        'en-US,en;q=0.9,zh;q=0.8'
    ]
    
    def __init__(self, cache_fingerprints: bool = True):
        """
        Initialize browser configurator
        
        Args:
            cache_fingerprints: Whether to cache generated fingerprints
        """
        self.cache_fingerprints = cache_fingerprints
        self.fingerprint_cache: Dict[str, BrowserFingerprint] = {}
        self.session_fingerprints: Dict[str, str] = {}
    
    def create_stealth_config(self,
                             anti_detection_level: str = "standard",
                             device_profile: DeviceProfile = DeviceProfile.DESKTOP_WINDOWS,
                             proxy_config: Optional[ProxyConfig] = None,
                             session_id: Optional[str] = None) -> BrowserConfig:
        """
        Create comprehensive stealth browser configuration
        
        Args:
            anti_detection_level: Level of anti-detection measures
            device_profile: Device simulation profile
            proxy_config: Proxy configuration
            session_id: Session identifier for fingerprint consistency
            
        Returns:
            Configured BrowserConfig object
        """
        
        # Generate or retrieve fingerprint
        fingerprint = self._get_session_fingerprint(session_id, device_profile)
        
        # Create base configuration
        config = BrowserConfig(
            browser_type=BrowserType.CHROMIUM.value,
            headless=True,
            viewport_width=fingerprint.viewport.width,
            viewport_height=fingerprint.viewport.height,
            user_agent=fingerprint.user_agent
        )
        
        # Apply anti-detection measures based on level
        if anti_detection_level == "minimal":
            config.extra_args = self._get_minimal_args()
        elif anti_detection_level == "standard":
            config.extra_args = self._get_standard_args(fingerprint)
        elif anti_detection_level == "aggressive":
            config.extra_args = self._get_aggressive_args(fingerprint)
        elif anti_detection_level == "stealth":
            config.extra_args = self._get_stealth_args(fingerprint)
        
        # Add proxy configuration
        if proxy_config:
            config.proxy = {
                'server': proxy_config.server,
                'username': proxy_config.username,
                'password': proxy_config.password
            }
            if proxy_config.bypass:
                config.proxy['bypass'] = ','.join(proxy_config.bypass)
        
        return config
    
    def _get_session_fingerprint(self,
                                session_id: Optional[str],
                                device_profile: DeviceProfile) -> BrowserFingerprint:
        """Get or generate fingerprint for session"""
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check if we have a cached fingerprint for this session
        if session_id in self.session_fingerprints:
            fingerprint_key = self.session_fingerprints[session_id]
            if fingerprint_key in self.fingerprint_cache:
                return self.fingerprint_cache[fingerprint_key]
        
        # Generate new fingerprint
        fingerprint = self._generate_realistic_fingerprint(device_profile)
        
        if self.cache_fingerprints:
            fingerprint_key = f"{device_profile.value}_{hash(str(fingerprint))}"
            self.fingerprint_cache[fingerprint_key] = fingerprint
            self.session_fingerprints[session_id] = fingerprint_key
        
        return fingerprint
    
    def _generate_realistic_fingerprint(self, device_profile: DeviceProfile) -> BrowserFingerprint:
        """Generate realistic browser fingerprint"""
        
        viewport = self.VIEWPORTS[device_profile]
        
        # Select appropriate user agent
        if device_profile == DeviceProfile.DESKTOP_WINDOWS:
            user_agent = random.choice(self.USER_AGENTS['windows_chrome'])
            platform = "Win32"
        elif device_profile == DeviceProfile.DESKTOP_MAC:
            user_agent = random.choice(self.USER_AGENTS['mac_chrome'] + self.USER_AGENTS['mac_safari'])
            platform = "MacIntel"
        elif device_profile == DeviceProfile.DESKTOP_LINUX:
            user_agent = random.choice(self.USER_AGENTS['windows_chrome'])  # Use Windows UA but Linux platform
            platform = "Linux x86_64"
        elif device_profile in [DeviceProfile.MOBILE_ANDROID, DeviceProfile.TABLET_ANDROID]:
            user_agent = random.choice(self.USER_AGENTS['mobile_android'])
            platform = "Linux armv8l"
        else:  # iOS devices
            user_agent = random.choice(self.USER_AGENTS['mobile_ios'])
            platform = "iPhone" if "iPhone" in user_agent else "iPad"
        
        # Generate screen resolution (larger than viewport)
        if viewport.is_mobile:
            screen_width = viewport.width
            screen_height = viewport.height + random.randint(50, 150)  # Account for browser UI
        else:
            screen_width = random.choice([1920, 2560, 1680, 1440])
            screen_height = random.choice([1080, 1440, 1050, 900])
        
        return BrowserFingerprint(
            user_agent=user_agent,
            viewport=viewport,
            timezone=random.choice(self.TIMEZONES),
            language=random.choice(self.LANGUAGES),
            platform=platform,
            screen_resolution=(screen_width, screen_height),
            color_depth=random.choice([24, 32]),
            pixel_ratio=viewport.device_scale_factor,
            hardware_concurrency=random.choice([4, 8, 12, 16]),
            memory=random.choice([4, 8, 16, 32]) * 1024,  # In MB
            webgl_vendor="Google Inc.",
            webgl_renderer=self._get_realistic_gpu()
        )
    
    def _get_realistic_gpu(self) -> str:
        """Get realistic GPU renderer string"""
        gpus = [
            "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)",
            "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)",
            "ANGLE (AMD, AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0, D3D11)",
            "Apple GPU",
            "Mali-G78 MC14",
            "Adreno (TM) 640"
        ]
        return random.choice(gpus)
    
    def _get_minimal_args(self) -> List[str]:
        """Minimal anti-detection arguments"""
        return [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu'
        ]
    
    def _get_standard_args(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Standard anti-detection arguments"""
        return [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--no-first-run',
            '--disable-default-apps',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection',
            f'--lang={fingerprint.language.split(",")[0]}',
            f'--timezone={fingerprint.timezone}'
        ]
    
    def _get_aggressive_args(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Aggressive anti-detection arguments"""
        return self._get_standard_args(fingerprint) + [
            '--disable-plugins',
            '--disable-javascript-harmony-shipping',
            '--disable-web-security',
            '--allow-running-insecure-content',
            '--disable-features=VizDisplayCompositor',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-field-trial-config',
            '--disable-back-forward-cache'
        ]
    
    def _get_stealth_args(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Maximum stealth arguments"""
        return self._get_aggressive_args(fingerprint) + [
            '--no-zygote',
            '--disable-background-networking',
            '--disable-background-mode',
            '--disable-client-side-phishing-detection',
            '--disable-component-extensions-with-background-pages',
            '--disable-default-apps',
            '--disable-hang-monitor',
            '--disable-prompt-on-repost',
            '--disable-sync',
            '--metrics-recording-only',
            '--no-report-upload',
            '--enable-automation=false',
            '--password-store=basic',
            '--use-mock-keychain'
        ]
    
    def create_mobile_config(self,
                           mobile_device: str = "Pixel 7",
                           anti_detection_level: str = "standard") -> BrowserConfig:
        """
        Create mobile-optimized browser configuration
        
        Args:
            mobile_device: Mobile device to simulate
            anti_detection_level: Anti-detection level
            
        Returns:
            Mobile-optimized BrowserConfig
        """
        
        device_profiles = {
            "Pixel 7": DeviceProfile.MOBILE_ANDROID,
            "iPhone 14": DeviceProfile.MOBILE_IOS,
            "iPad Air": DeviceProfile.TABLET_IOS,
            "Galaxy Tab": DeviceProfile.TABLET_ANDROID
        }
        
        profile = device_profiles.get(mobile_device, DeviceProfile.MOBILE_ANDROID)
        
        return self.create_stealth_config(
            anti_detection_level=anti_detection_level,
            device_profile=profile
        )
    
    def create_proxy_rotated_config(self,
                                   proxy_list: List[str],
                                   anti_detection_level: str = "standard") -> BrowserConfig:
        """
        Create configuration with proxy rotation
        
        Args:
            proxy_list: List of proxy servers
            anti_detection_level: Anti-detection level
            
        Returns:
            Proxy-enabled BrowserConfig
        """
        
        if not proxy_list:
            return self.create_stealth_config(anti_detection_level=anti_detection_level)
        
        # Select random proxy
        proxy_server = random.choice(proxy_list)
        
        proxy_config = ProxyConfig(server=proxy_server)
        
        return self.create_stealth_config(
            anti_detection_level=anti_detection_level,
            proxy_config=proxy_config
        )
    
    def inject_fingerprint_script(self, fingerprint: BrowserFingerprint) -> str:
        """
        Generate JavaScript code to override browser fingerprint
        
        Args:
            fingerprint: Browser fingerprint to inject
            
        Returns:
            JavaScript code string
        """
        
        script = f"""
        // Override navigator properties
        Object.defineProperty(navigator, 'platform', {{
            get: () => '{fingerprint.platform}'
        }});
        
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {fingerprint.hardware_concurrency}
        }});
        
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {fingerprint.memory // 1024}
        }});
        
        Object.defineProperty(navigator, 'language', {{
            get: () => '{fingerprint.language.split(",")[0]}'
        }});
        
        Object.defineProperty(navigator, 'languages', {{
            get: () => {json.dumps(fingerprint.language.split(","))}
        }});
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {{
            get: () => {fingerprint.screen_resolution[0]}
        }});
        
        Object.defineProperty(screen, 'height', {{
            get: () => {fingerprint.screen_resolution[1]}
        }});
        
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => {fingerprint.color_depth}
        }});
        
        Object.defineProperty(window, 'devicePixelRatio', {{
            get: () => {fingerprint.pixel_ratio}
        }});
        
        // Override WebGL properties
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) {{
                return '{fingerprint.webgl_vendor}';
            }}
            if (parameter === 37446) {{
                return '{fingerprint.webgl_renderer}';
            }}
            return getParameter.call(this, parameter);
        }};
        
        // Override timezone
        const originalGetTimezoneOffset = Date.prototype.getTimezoneOffset;
        Date.prototype.getTimezoneOffset = function() {{
            return {self._get_timezone_offset(fingerprint.timezone)};
        }};
        
        // Remove automation indicators
        Object.defineProperty(navigator, 'webdriver', {{
            get: () => undefined
        }});
        
        // Override plugin detection
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {{
                return [
                    {{
                        name: 'Chrome PDF Plugin',
                        filename: 'internal-pdf-viewer',
                        description: 'Portable Document Format'
                    }}
                ];
            }}
        }});
        """
        
        return script
    
    def _get_timezone_offset(self, timezone: str) -> int:
        """Get timezone offset in minutes"""
        timezone_offsets = {
            'America/New_York': 300,
            'America/Chicago': 360,
            'America/Denver': 420,
            'America/Los_Angeles': 480,
            'Europe/London': 0,
            'Europe/Paris': -60,
            'Europe/Berlin': -60,
            'Europe/Rome': -60,
            'Asia/Tokyo': -540,
            'Asia/Shanghai': -480,
            'Asia/Seoul': -540,
            'Australia/Sydney': -660
        }
        return timezone_offsets.get(timezone, 0)
    
    def validate_config(self, config: BrowserConfig) -> Dict[str, Any]:
        """
        Validate browser configuration
        
        Args:
            config: Browser configuration to validate
            
        Returns:
            Validation results
        """
        
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check user agent
        if not config.user_agent:
            results['errors'].append("User agent is required")
            results['valid'] = False
        
        # Check viewport
        if config.viewport_width <= 0 or config.viewport_height <= 0:
            results['errors'].append("Invalid viewport dimensions")
            results['valid'] = False
        
        # Check for common detection indicators
        if config.extra_args:
            if '--enable-automation' in config.extra_args:
                results['warnings'].append("Automation flag detected - may increase detection risk")
        
        return results
    
    def get_config_summary(self, config: BrowserConfig) -> Dict[str, Any]:
        """
        Get summary of browser configuration
        
        Args:
            config: Browser configuration
            
        Returns:
            Configuration summary
        """
        
        return {
            'browser_type': config.browser_type,
            'headless': config.headless,
            'viewport': f"{config.viewport_width}x{config.viewport_height}",
            'user_agent_browser': self._extract_browser_from_ua(config.user_agent),
            'proxy_enabled': bool(config.proxy),
            'stealth_args_count': len(config.extra_args) if config.extra_args else 0,
            'anti_detection_level': self._assess_stealth_level(config)
        }
    
    def _extract_browser_from_ua(self, user_agent: str) -> str:
        """Extract browser name from user agent"""
        if not user_agent:
            return "Unknown"
        
        ua_lower = user_agent.lower()
        if 'chrome' in ua_lower:
            return "Chrome"
        elif 'firefox' in ua_lower:
            return "Firefox"
        elif 'safari' in ua_lower and 'chrome' not in ua_lower:
            return "Safari"
        else:
            return "Unknown"
    
    def _assess_stealth_level(self, config: BrowserConfig) -> str:
        """Assess stealth level based on configuration"""
        if not config.extra_args:
            return "minimal"
        
        arg_count = len(config.extra_args)
        
        if arg_count < 5:
            return "minimal"
        elif arg_count < 10:
            return "standard"
        elif arg_count < 20:
            return "aggressive"
        else:
            return "stealth"