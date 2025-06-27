# 📐 Comprehensive Class Diagrams - eBay Jewelry Scraper

## 🎯 Overview

This document provides detailed class diagrams and component interaction specifications for the eBay Jewelry Scraper architecture. The system is designed with modular components that work together to provide robust, scalable, and intelligent jewelry data extraction.

## 🏗️ Core Architecture Classes

### Main Scraper Engine Classes

```mermaid
classDiagram
    class EbayJewelryCrawler {
        +config: ScraperConfig
        +session_manager: SessionManager
        +anti_detection: AntiDetectionSystem
        +rate_limiter: AdaptiveRateLimiter
        +browser_pool: BrowserPool
        +data_extractor: JewelryDataExtractor
        +image_processor: ImageProcessor
        +validator: DataValidator
        +progress_tracker: ProgressTracker
        +error_manager: ErrorManager
        
        +scrape_category(category: JewelryCategory, params: SearchParameters) ScrapingResult
        +scrape_listing(url: str) Optional[JewelryListing]
        +batch_scrape(urls: List[str]) List[JewelryListing]
        +resume_session(session_id: str) void
        +pause_session(session_id: str) void
        +get_session_status(session_id: str) SessionStatus
        +optimize_performance() void
        +get_statistics() CrawlerStatistics
    }

    class ScraperConfig {
        +database: DatabaseConfig
        +anti_detection: AntiDetectionConfig
        +rate_limiting: RateLimitConfig
        +extraction: ExtractionConfig
        +image_processing: ImageProcessingConfig
        +monitoring: MonitoringConfig
        +performance: PerformanceConfig
        
        +load_from_file(path: str) ScraperConfig
        +save_to_file(path: str) void
        +validate() bool
        +merge(other: ScraperConfig) ScraperConfig
    }

    class SessionManager {
        +active_sessions: Dict[str, ScrapingSession]
        +session_store: SessionStore
        +state_persister: StatePersister
        +recovery_manager: RecoveryManager
        +checkpoint_manager: CheckpointManager
        
        +create_session(config: SessionConfiguration) ScrapingSession
        +save_session_state(session: ScrapingSession) void
        +recover_session(session_id: str) Optional[ScrapingSession]
        +cleanup_session(session_id: str) void
        +list_active_sessions() List[ScrapingSession]
        +get_session_metrics(session_id: str) SessionMetrics
    }

    class BrowserPool {
        +pool_size: int
        +available_browsers: Queue[Browser]
        +busy_browsers: Set[Browser]
        +browser_configs: List[BrowserConfig]
        +health_checker: BrowserHealthChecker
        
        +get_browser() Browser
        +return_browser(browser: Browser) void
        +create_browser() Browser
        +destroy_browser(browser: Browser) void
        +health_check() void
        +resize_pool(new_size: int) void
        +get_pool_statistics() PoolStatistics
    }

    EbayJewelryCrawler --> ScraperConfig
    EbayJewelryCrawler --> SessionManager
    EbayJewelryCrawler --> BrowserPool
    SessionManager --> ScrapingSession
```

### Anti-Detection System Classes

```mermaid
classDiagram
    class AntiDetectionSystem {
        +user_agent_manager: UserAgentManager
        +proxy_rotator: ProxyRotator
        +fingerprint_randomizer: BrowserFingerprintRandomizer
        +request_pattern_manager: RequestPatternManager
        +captcha_solver: CaptchaSolver
        +detection_analyzer: DetectionAnalyzer
        +strategy_adapter: StrategyAdapter
        
        +prepare_request(type: RequestType, mobile: bool) Dict[str, Any]
        +execute_request(page: Page, url: str, config: Dict) Tuple[Response, str, DetectionEvent]
        +handle_detection(event: DetectionEvent) void
        +adapt_strategy() void
        +get_statistics() Dict[str, Any]
    }

    class UserAgentManager {
        +desktop_agents: List[str]
        +mobile_agents: List[str]
        +agent_metrics: Dict[str, Dict]
        +rotation_frequency: int
        +current_agent: str
        +request_count: int
        
        +get_user_agent(mobile: bool) str
        +report_success(user_agent: str, success: bool) void
        +rotate_user_agent(mobile: bool) void
        +get_best_agents(count: int) List[str]
        +block_agent(user_agent: str) void
        +unblock_agent(user_agent: str) void
    }

    class ProxyRotator {
        +proxies: List[ProxyConfig]
        +current_index: int
        +health_checker: ProxyHealthChecker
        +failover_manager: ProxyFailoverManager
        +rotation_frequency: int
        
        +get_proxy() Optional[ProxyConfig]
        +report_proxy_result(proxy: ProxyConfig, success: bool, response_time: float) void
        +health_check_proxies() void
        +rotate_proxy() void
        +ban_proxy(proxy: ProxyConfig) void
        +get_healthy_proxies() List[ProxyConfig]
    }

    class BrowserFingerprintRandomizer {
        +viewport_ranges: Dict[str, List[Tuple]]
        +timezones: List[str]
        +languages: List[str]
        +confidence_weights: Dict[str, float]
        
        +generate_fingerprint(user_agent: str, mobile: bool) BrowserFingerprint
        +apply_fingerprint(page: Page, fingerprint: BrowserFingerprint) void
        +randomize_properties() Dict[str, Any]
    }

    class RequestPatternManager {
        +min_delay: float
        +max_delay: float
        +request_history: deque
        +human_simulation: bool
        +burst_protection: bool
        
        +calculate_delay(request_type: RequestType) float
        +record_request(type: RequestType, delay: float) void
        +simulate_human_interactions(page: Page) void
        +analyze_patterns() PatternAnalysis
    }

    class DetectionAnalyzer {
        +captcha_indicators: List[str]
        +rate_limit_indicators: List[str]
        +bot_detection_indicators: List[str]
        +response_time_threshold: float
        
        +analyze_response(response: Response, content: str, time: float) DetectionEvent
        +classify_detection_level(indicators: List[str]) DetectionLevel
        +extract_detection_signals(content: str) List[str]
    }

    AntiDetectionSystem --> UserAgentManager
    AntiDetectionSystem --> ProxyRotator
    AntiDetectionSystem --> BrowserFingerprintRandomizer
    AntiDetectionSystem --> RequestPatternManager
    AntiDetectionSystem --> DetectionAnalyzer
```

### Data Extraction Pipeline Classes

```mermaid
classDiagram
    class JewelryDataExtractor {
        +selector_manager: SelectorManager
        +categorizer: JewelryCategorizer
        +price_parser: PriceParser
        +spec_extractor: SpecificationExtractor
        +validator: DataValidator
        +ml_models: MLModelManager
        
        +extract_listing_data(html: str, url: str) JewelryListing
        +extract_images(html: str) List[ImageMetadata]
        +validate_extraction(listing: JewelryListing) ValidationResult
        +enhance_with_ai(listing: JewelryListing) JewelryListing
        +calculate_quality_score(listing: JewelryListing) float
    }

    class SelectorManager {
        +selectors: Dict[str, List[SelectorPattern]]
        +selector_success_rates: Dict[str, Dict[str, float]]
        +selector_usage_counts: Dict[str, Dict[str, int]]
        
        +extract_data(soup: BeautifulSoup, data_type: str) List[ExtractionResult]
        +optimize_selectors() void
        +add_selector_pattern(data_type: str, pattern: SelectorPattern) void
        +get_best_selectors(data_type: str) List[SelectorPattern]
        +update_selector_success_rate(data_type: str, selector: str, success: bool) void
    }

    class JewelryCategorizer {
        +category_keywords: Dict[JewelryCategory, Dict[str, float]]
        +material_keywords: Dict[JewelryMaterial, Dict[str, float]]
        +gemstone_keywords: Dict[str, float]
        +lemmatizer: WordNetLemmatizer
        +stop_words: Set[str]
        
        +categorize_jewelry(title: str, description: str) ExtractionResult
        +extract_material(title: str, description: str) ExtractionResult
        +extract_gemstones(title: str, description: str) List[ExtractionResult]
        +analyze_text_tokens(text: str) List[str]
        +calculate_category_confidence(matches: List[str], category: JewelryCategory) float
    }

    class PriceParser {
        +price_patterns: List[Tuple[str, str, float]]
        +price_indicators: List[str]
        
        +parse_price(text: str, context: str) ExtractionResult
        +validate_price_range(price: float) bool
        +extract_currency(text: str) str
        +normalize_price_format(text: str) str
    }

    class SpecificationExtractor {
        +field_mappings: Dict[str, List[str]]
        +confidence_weights: Dict[str, float]
        
        +extract_specifications(soup: BeautifulSoup) Dict[str, ExtractionResult]
        +extract_from_tables(soup: BeautifulSoup) Dict[str, ExtractionResult]
        +extract_from_definition_lists(soup: BeautifulSoup) Dict[str, ExtractionResult]
        +extract_from_labeled_elements(soup: BeautifulSoup) Dict[str, ExtractionResult]
        +map_field_name(label: str) Optional[str]
    }

    class DataValidator {
        +validation_rules: ValidationRuleEngine
        +quality_metrics: QualityMetricsCalculator
        +anomaly_detector: AnomalyDetector
        +ml_validator: MLDataValidator
        
        +validate_listing(listing: JewelryListing) ValidationResult
        +validate_batch(listings: List[JewelryListing]) BatchValidationResult
        +suggest_corrections(listing: JewelryListing, result: ValidationResult) List[Correction]
        +calculate_data_quality_score(listing: JewelryListing) float
    }

    JewelryDataExtractor --> SelectorManager
    JewelryDataExtractor --> JewelryCategorizer
    JewelryDataExtractor --> PriceParser
    JewelryDataExtractor --> SpecificationExtractor
    JewelryDataExtractor --> DataValidator
```

### Error Handling System Classes

```mermaid
classDiagram
    class ErrorManager {
        +classifier: ErrorClassifier
        +retry_configs: Dict[ErrorCategory, RetryConfig]
        +circuit_breakers: Dict[str, CircuitBreaker]
        +rate_limiters: Dict[str, AdaptiveRateLimiter]
        +error_records: List[ErrorRecord]
        +error_statistics: Dict[str, Any]
        
        +handle_error(error: Exception, context: Dict, operation: str) ErrorRecord
        +execute_with_protection(func: Callable, operation: str, context: Dict) Any
        +get_error_statistics() Dict[str, Any]
        +cleanup_old_errors(days_old: int) void
    }

    class ErrorClassifier {
        +classification_patterns: Dict[ErrorCategory, List[str]]
        
        +classify_error(error: Exception, context: Dict) Tuple[ErrorCategory, ErrorSeverity]
        +determine_severity(error: Exception, category: ErrorCategory) ErrorSeverity
        +extract_error_signals(error: Exception) List[str]
    }

    class RetryHandler {
        +config: RetryConfig
        
        +execute_with_retry(func: Callable) Any
        +should_retry(exception: Exception, attempt: int) bool
        +calculate_delay(attempt: int) float
        +apply_jitter(delay: float) float
    }

    class CircuitBreaker {
        +config: CircuitBreakerConfig
        +state: CircuitBreakerState
        +failure_count: int
        +success_count: int
        +last_failure_time: float
        
        +call(func: Callable) Any
        +record_success() void
        +record_failure() void
        +can_execute() bool
        +get_state() CircuitBreakerState
    }

    class AdaptiveRateLimiter {
        +config: RateLimitConfig
        +current_rate: float
        +token_bucket: TokenBucket
        +sliding_window: SlidingWindowCounter
        +success_count: int
        +error_count: int
        
        +acquire() bool
        +report_result(success: bool) void
        +adjust_rate() void
        +get_current_rate() float
    }

    class TokenBucket {
        +rate: float
        +capacity: int
        +tokens: float
        +last_update: float
        
        +acquire(tokens: int) bool
        +wait_for_token(tokens: int) float
        +refill() void
    }

    ErrorManager --> ErrorClassifier
    ErrorManager --> RetryHandler
    ErrorManager --> CircuitBreaker
    ErrorManager --> AdaptiveRateLimiter
    AdaptiveRateLimiter --> TokenBucket
```

## 📊 Data Models and Storage Classes

### Core Data Models

```mermaid
classDiagram
    class JewelryListing {
        +listing_id: str
        +url: str
        +title: str
        +price: Optional[float]
        +original_price: Optional[float]
        +category: JewelryCategory
        +material: JewelryMaterial
        +seller_info: SellerInfo
        +specifications: List[JewelrySpecification]
        +images: List[JewelryImage]
        +quality_metrics: QualityMetrics
        +validation_errors: List[str]
        
        +calculate_completeness_score() float
        +validate() ValidationResult
        +to_dict() Dict[str, Any]
        +from_dict(data: Dict) JewelryListing
        +enhance_data() void
        +get_image_urls() List[str]
    }

    class JewelryImage {
        +image_id: str
        +listing_id: str
        +original_url: str
        +local_path: Optional[str]
        +image_type: ImageType
        +sequence_order: int
        +file_size: Optional[int]
        +dimensions: Tuple[int, int]
        +quality_score: Optional[float]
        +metadata: Dict[str, Any]
        
        +download() bool
        +process() ProcessingResult
        +calculate_quality() float
        +generate_thumbnail() str
        +extract_colors() List[str]
    }

    class JewelrySpecification {
        +spec_id: str
        +listing_id: str
        +attribute_name: str
        +attribute_value: str
        +confidence_score: float
        +source_section: Optional[str]
        +standardized_name: Optional[str]
        +standardized_value: Optional[str]
        
        +standardize() void
        +validate() bool
        +merge_with(other: JewelrySpecification) JewelrySpecification
    }

    class ScrapingSession {
        +session_id: str
        +status: ScrapingStatus
        +progress: ProgressMetrics
        +configuration: SessionConfiguration
        +statistics: SessionStatistics
        +error_log: List[ErrorRecord]
        +checkpoints: List[SessionCheckpoint]
        
        +start() void
        +pause() void
        +resume() void
        +stop() void
        +save_checkpoint() void
        +restore_from_checkpoint(checkpoint_id: str) void
        +get_progress() ProgressMetrics
    }

    JewelryListing --> JewelryImage
    JewelryListing --> JewelrySpecification
    ScrapingSession --> JewelryListing
```

### Storage and Database Classes

```mermaid
classDiagram
    class DatabaseManager {
        +db_path: Path
        +connection_pool: ConnectionPool
        +query_optimizer: QueryOptimizer
        +backup_manager: BackupManager
        
        +create_tables() void
        +insert_listing(listing: JewelryListing) bool
        +update_listing(listing: JewelryListing) bool
        +get_listing(listing_id: str) Optional[JewelryListing]
        +search_listings(criteria: SearchCriteria) List[JewelryListing]
        +delete_old_listings(days: int) int
        +backup_database() str
        +restore_database(backup_path: str) bool
        +get_statistics() DatabaseStatistics
    }

    class CacheManager {
        +cache_type: CacheType
        +max_size: int
        +ttl: int
        +storage: Dict[str, CacheEntry]
        +hit_count: int
        +miss_count: int
        
        +get(key: str) Optional[Any]
        +set(key: str, value: Any, ttl: Optional[int]) void
        +delete(key: str) bool
        +clear() void
        +cleanup_expired() void
        +get_cache_statistics() CacheStatistics
    }

    class FileStorageManager {
        +base_path: Path
        +compression_enabled: bool
        +encryption_enabled: bool
        +retention_policy: RetentionPolicy
        
        +store_file(file_path: str, content: bytes) str
        +retrieve_file(file_id: str) Optional[bytes]
        +delete_file(file_id: str) bool
        +list_files(pattern: str) List[str]
        +cleanup_old_files(days: int) int
        +get_storage_statistics() StorageStatistics
    }

    class MetricsCollector {
        +metrics_store: MetricsStore
        +aggregator: MetricsAggregator
        +exporters: List[MetricsExporter]
        
        +record_metric(name: str, value: float, tags: Dict) void
        +increment_counter(name: str, tags: Dict) void
        +set_gauge(name: str, value: float, tags: Dict) void
        +record_histogram(name: str, value: float, tags: Dict) void
        +export_metrics() void
        +get_metric_summary() MetricsSummary
    }

    DatabaseManager --> JewelryListing
    CacheManager --> CacheEntry
    FileStorageManager --> JewelryImage
```

## 🔄 Component Interaction Flows

### Complete Scraping Operation Flow

```mermaid
sequenceDiagram
    participant CLI as CLI Interface
    participant SM as Session Manager
    participant EC as EbayJewelryCrawler
    participant ADS as Anti-Detection System
    participant BP as Browser Pool
    participant DE as Data Extractor
    participant IP as Image Processor
    participant DV as Data Validator
    participant DB as Database Manager
    participant EM as Error Manager
    participant PT as Progress Tracker

    CLI->>SM: create_session(config)
    SM-->>CLI: session_id
    
    CLI->>EC: scrape_category(category, params)
    EC->>SM: get_session(session_id)
    EC->>ADS: prepare_request(SEARCH_PAGE)
    ADS-->>EC: request_config
    
    EC->>BP: get_browser()
    BP-->>EC: browser_instance
    
    loop For each search page
        EC->>ADS: execute_request(page, url, config)
        ADS->>EM: execute_with_protection(navigate)
        
        alt Success
            ADS-->>EC: (response, content, detection_event)
            EC->>DE: extract_listing_urls(content)
            DE-->>EC: listing_urls
            
            loop For each listing URL
                EC->>ADS: prepare_request(LISTING_PAGE)
                EC->>ADS: execute_request(page, listing_url, config)
                
                alt Success
                    ADS-->>EC: (response, listing_content, event)
                    EC->>DE: extract_listing_data(listing_content, url)
                    DE-->>EC: jewelry_listing
                    
                    EC->>DV: validate_listing(jewelry_listing)
                    DV-->>EC: validation_result
                    
                    alt Valid listing
                        EC->>IP: process_images(jewelry_listing.images)
                        IP-->>EC: processed_images
                        
                        EC->>DB: insert_listing(jewelry_listing)
                        DB-->>EC: success
                        
                        EC->>PT: update_progress(session_id, listing)
                        PT-->>CLI: progress_notification
                    else Invalid listing
                        EC->>PT: record_failure(session_id, validation_errors)
                    end
                    
                else Request failed
                    EC->>EM: handle_error(error, context)
                    EM-->>EC: error_record
                end
            end
            
        else Search page failed
            EC->>EM: handle_error(error, context)
            EM-->>EC: error_record
        end
    end
    
    EC->>BP: return_browser(browser_instance)
    EC->>SM: complete_session(session_id)
    EC-->>CLI: scraping_result
```

### Error Recovery Flow

```mermaid
sequenceDiagram
    participant EC as EbayJewelryCrawler
    participant EM as Error Manager
    participant CB as Circuit Breaker
    participant RH as Retry Handler
    participant ADS as Anti-Detection System
    participant RL as Rate Limiter

    EC->>EM: execute_with_protection(operation)
    EM->>CB: call(operation)
    
    alt Circuit Breaker CLOSED
        CB->>RH: execute_with_retry(operation)
        
        loop Retry attempts
            RH->>EC: execute_operation()
            
            alt Operation fails
                EC-->>RH: exception
                RH->>EM: classify_error(exception)
                EM-->>RH: (category, severity)
                
                alt Should retry
                    RH->>RH: calculate_delay(attempt)
                    RH->>RH: wait(delay)
                    
                    alt Rate limited error
                        RH->>ADS: adapt_strategy()
                        ADS->>RL: decrease_rate()
                    end
                else No retry
                    RH-->>CB: final_exception
                    CB->>CB: record_failure()
                    CB-->>EM: exception
                end
            else Operation succeeds
                RH-->>CB: result
                CB->>CB: record_success()
                CB-->>EM: result
            end
        end
        
    else Circuit Breaker OPEN
        CB-->>EM: CircuitBreakerOpenException
        EM->>EM: handle_circuit_breaker_open()
        EM-->>EC: circuit_breaker_exception
    end
```

### Adaptive Rate Limiting Flow

```mermaid
sequenceDiagram
    participant EC as EbayJewelryCrawler
    participant ARL as Adaptive Rate Limiter
    participant TB as Token Bucket
    participant SW as Sliding Window
    participant EM as Error Manager

    EC->>ARL: acquire()
    
    alt Token bucket strategy
        ARL->>TB: acquire(1)
        
        alt Tokens available
            TB-->>ARL: true
            ARL-->>EC: permission_granted
        else No tokens
            TB->>TB: calculate_wait_time()
            TB-->>ARL: wait_time
            ARL->>ARL: sleep(wait_time)
            ARL->>TB: acquire(1)
            TB-->>ARL: true
            ARL-->>EC: permission_granted
        end
        
    else Sliding window strategy
        ARL->>SW: can_proceed()
        SW-->>ARL: can_proceed
        
        alt Can proceed
            ARL->>SW: record_request()
            ARL-->>EC: permission_granted
        else Rate exceeded
            ARL->>ARL: sleep(window_remainder)
            ARL-->>EC: permission_granted
        end
    end
    
    EC->>EC: execute_request()
    
    alt Request successful
        EC->>ARL: report_result(success=true)
        ARL->>ARL: increment_success_count()
        
        Note over ARL: Check if rate can be increased
        ARL->>ARL: maybe_adjust_rate()
        
    else Request failed
        EC->>ARL: report_result(success=false)
        ARL->>ARL: increment_error_count()
        
        alt High error rate
            ARL->>ARL: decrease_rate()
            ARL->>TB: update_rate(new_rate)
        end
    end
```

## 🎛️ Configuration and Factory Classes

### Configuration Management

```mermaid
classDiagram
    class ConfigurationManager {
        +config_path: Path
        +environment_overrides: Dict[str, Any]
        +validation_schema: Dict[str, Any]
        
        +load_configuration() ScraperConfig
        +save_configuration(config: ScraperConfig) void
        +validate_configuration(config: ScraperConfig) ValidationResult
        +merge_environments(configs: List[ScraperConfig]) ScraperConfig
        +watch_config_changes() void
        +reload_configuration() void
    }

    class ComponentFactory {
        +config: ScraperConfig
        +component_registry: Dict[str, Type]
        +instance_cache: Dict[str, Any]
        
        +create_crawler() EbayJewelryCrawler
        +create_anti_detection_system() AntiDetectionSystem
        +create_data_extractor() JewelryDataExtractor
        +create_error_manager() ErrorManager
        +create_database_manager() DatabaseManager
        +register_component(name: str, component_type: Type) void
        +get_component(name: str) Any
    }

    class DependencyInjector {
        +bindings: Dict[Type, Type]
        +singletons: Dict[Type, Any]
        +factories: Dict[Type, Callable]
        
        +bind(interface: Type, implementation: Type) void
        +bind_singleton(interface: Type, instance: Any) void
        +bind_factory(interface: Type, factory: Callable) void
        +get(interface: Type) Any
        +create_instance(target_type: Type) Any
    }

    ConfigurationManager --> ScraperConfig
    ComponentFactory --> ConfigurationManager
    ComponentFactory --> DependencyInjector
```

## 📈 Monitoring and Observability Classes

### Monitoring System

```mermaid
classDiagram
    class ProgressTracker {
        +session_metrics: Dict[str, SessionMetrics]
        +event_publisher: EventPublisher
        +metrics_collector: MetricsCollector
        +notification_manager: NotificationManager
        +dashboard_updater: DashboardUpdater
        
        +track_session_progress(session: ScrapingSession) void
        +publish_milestone(milestone: ProgressMilestone) void
        +generate_progress_report(session_id: str) ProgressReport
        +calculate_eta(session: ScrapingSession) timedelta
        +get_performance_metrics() PerformanceMetrics
    }

    class HealthChecker {
        +component_health_checks: Dict[str, HealthCheck]
        +system_health_checks: List[SystemHealthCheck]
        +health_status: HealthStatus
        +check_interval: int
        
        +check_all_components() HealthReport
        +check_component(component_name: str) ComponentHealth
        +register_health_check(name: str, check: HealthCheck) void
        +start_monitoring() void
        +stop_monitoring() void
        +get_health_summary() HealthSummary
    }

    class AlertManager {
        +alert_rules: List[AlertRule]
        +notification_channels: List[NotificationChannel]
        +alert_history: List[Alert]
        +suppression_rules: List[SuppressionRule]
        
        +evaluate_alerts(metrics: MetricsSnapshot) void
        +send_alert(alert: Alert) void
        +suppress_alert(alert: Alert, duration: timedelta) void
        +acknowledge_alert(alert_id: str) void
        +get_active_alerts() List[Alert]
    }

    class DashboardService {
        +websocket_connections: Set[WebSocketConnection]
        +data_aggregator: DataAggregator
        +chart_generators: Dict[str, ChartGenerator]
        +real_time_updates: bool
        
        +serve_dashboard() void
        +broadcast_update(data: Dict[str, Any]) void
        +generate_chart(chart_type: str, data: Any) Chart
        +handle_websocket_connection(connection: WebSocketConnection) void
        +get_dashboard_data() DashboardData
    }

    ProgressTracker --> MetricsCollector
    HealthChecker --> ProgressTracker
    AlertManager --> HealthChecker
    DashboardService --> AlertManager
```

This comprehensive class diagram specification provides the complete architectural foundation for the eBay Jewelry Scraper system, showing all major components, their relationships, and interaction patterns for robust, scalable, and maintainable jewelry data extraction.