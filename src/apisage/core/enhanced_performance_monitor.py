"""
Enhanced Performance Monitor with OpenTelemetry Integration
Provides comprehensive timing instrumentation for every step of the API cycle
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import asynccontextmanager, contextmanager
import json
import statistics
from functools import wraps

import structlog

# OpenTelemetry imports with fallbacks
try:
    from opentelemetry import trace, metrics
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OPENTELEMETRY_AVAILABLE = True
except ImportError as e:
    OPENTELEMETRY_AVAILABLE = False
    # Mock classes for when OpenTelemetry is not available
    class MockTracer:
        def start_span(self, *args, **kwargs):
            return MockSpan()

    class MockSpan:
        def set_attribute(self, *args, **kwargs): pass
        def end(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class MockMeter:
        def create_histogram(self, *args, **kwargs): return MockMetric()
        def create_counter(self, *args, **kwargs): return MockMetric()

    class MockMetric:
        def record(self, *args, **kwargs): pass
        def add(self, *args, **kwargs): pass

logger = structlog.get_logger()


@dataclass
class StepTiming:
    """Detailed timing for each processing step"""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['StepTiming'] = field(default_factory=list)

    def complete(self) -> float:
        """Mark step as complete and return duration"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self.duration_ms


@dataclass
class RequestProfile:
    """Complete performance profile for a request"""
    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    steps: List[StepTiming] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def analyze_bottlenecks(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        for step in self.steps:
            if step.duration_ms and step.duration_ms > threshold_ms:
                bottlenecks.append({
                    "step": step.step_name,
                    "duration_ms": step.duration_ms,
                    "percentage": (step.duration_ms / self.total_duration_ms * 100) if self.total_duration_ms else 0,
                    "metadata": step.metadata
                })
        self.bottlenecks = sorted(bottlenecks, key=lambda x: x["duration_ms"], reverse=True)
        return self.bottlenecks


class EnhancedPerformanceMonitor:
    """
    Advanced performance monitoring with OpenTelemetry integration
    Tracks detailed timing for every step in the request cycle
    """

    def __init__(
        self,
        service_name: str = "apisage",
        otlp_endpoint: Optional[str] = None,
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_profiling: bool = True
    ):
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_profiling = enable_profiling

        # Initialize OpenTelemetry
        if enable_tracing or enable_metrics:
            self._setup_opentelemetry(otlp_endpoint)

        # Request profiles storage
        self.active_profiles: Dict[str, RequestProfile] = {}
        self.completed_profiles: List[RequestProfile] = []
        self.max_completed_profiles = 1000

        # Performance thresholds for alerting
        self.thresholds = {
            "total_request_ms": 5000,
            "llm_call_ms": 3000,
            "db_query_ms": 100,
            "cache_lookup_ms": 10,
            "embedding_ms": 500,
            "context_assembly_ms": 200
        }

        # Metrics aggregation
        self.aggregated_metrics = {
            "request_count": 0,
            "total_duration_sum": 0,
            "step_durations": {},
            "error_count": 0
        }

    def _setup_opentelemetry(self, otlp_endpoint: Optional[str]):
        """Configure OpenTelemetry with OTLP exporters"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, using mock tracer and metrics")
            self.tracer = MockTracer()
            self.meter = MockMeter()
            return

        try:
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: "3.0.0",
            })

            if self.enable_tracing:
                # Setup tracing
                tracer_provider = TracerProvider(resource=resource)
                if otlp_endpoint:
                    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                    span_processor = BatchSpanProcessor(span_exporter)
                    tracer_provider.add_span_processor(span_processor)
                trace.set_tracer_provider(tracer_provider)
                self.tracer = trace.get_tracer(__name__)

                # Auto-instrument libraries
                try:
                    FastAPIInstrumentor.instrument()
                    HTTPXClientInstrumentor.instrument()
                except Exception as e:
                    logger.warning(f"Failed to auto-instrument: {e}")

            if self.enable_metrics:
                # Setup metrics
                if otlp_endpoint:
                    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
                    metric_reader = PeriodicExportingMetricReader(metric_exporter)
                    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                else:
                    meter_provider = MeterProvider(resource=resource)
                metrics.set_meter_provider(meter_provider)
                self.meter = metrics.get_meter(__name__)

                # Create metrics
                self._setup_metrics()
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
            self.tracer = MockTracer()
            self.meter = MockMeter()

    def _setup_metrics(self):
        """Setup OpenTelemetry metrics"""
        self.request_duration = self.meter.create_histogram(
            name="request_duration_ms",
            description="Request duration in milliseconds",
            unit="ms"
        )

        self.step_duration = self.meter.create_histogram(
            name="step_duration_ms",
            description="Individual step duration in milliseconds",
            unit="ms"
        )

        self.cache_hit_counter = self.meter.create_counter(
            name="cache_hits",
            description="Number of cache hits"
        )

        self.cache_miss_counter = self.meter.create_counter(
            name="cache_misses",
            description="Number of cache misses"
        )

        self.error_counter = self.meter.create_counter(
            name="errors",
            description="Number of errors"
        )

        self.token_counter = self.meter.create_counter(
            name="tokens_used",
            description="Total tokens used"
        )

    @contextmanager
    def profile_request(self, request_id: str, endpoint: str, method: str = "POST"):
        """Context manager for profiling entire request"""
        profile = RequestProfile(
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            start_time=time.time()
        )
        self.active_profiles[request_id] = profile

        span = None
        if self.enable_tracing:
            span = self.tracer.start_span(f"{method} {endpoint}")
            span.set_attribute("request.id", request_id)

        try:
            yield profile
        finally:
            profile.end_time = time.time()
            profile.total_duration_ms = (profile.end_time - profile.start_time) * 1000
            profile.analyze_bottlenecks()

            # Record metrics
            if self.enable_metrics:
                self.request_duration.record(
                    profile.total_duration_ms,
                    {"endpoint": endpoint, "method": method}
                )

            # Store completed profile
            self.completed_profiles.append(profile)
            if len(self.completed_profiles) > self.max_completed_profiles:
                self.completed_profiles.pop(0)

            # Remove from active
            del self.active_profiles[request_id]

            # End span
            if span:
                span.set_attribute("duration.ms", profile.total_duration_ms)
                span.end()

            # Log summary
            logger.info(
                "Request completed",
                request_id=request_id,
                endpoint=endpoint,
                duration_ms=profile.total_duration_ms,
                bottlenecks=profile.bottlenecks[:3]  # Top 3 bottlenecks
            )

    @contextmanager
    def time_step(self, request_id: str, step_name: str, **metadata):
        """Context manager for timing individual steps"""
        profile = self.active_profiles.get(request_id)
        if not profile:
            # Fallback if no active profile
            yield
            return

        step = StepTiming(
            step_name=step_name,
            start_time=time.time(),
            metadata=metadata
        )
        profile.steps.append(step)

        span = None
        if self.enable_tracing:
            span = self.tracer.start_span(step_name)
            for key, value in metadata.items():
                span.set_attribute(f"step.{key}", str(value))

        try:
            yield step
        finally:
            duration = step.complete()

            # Record step metric
            if self.enable_metrics:
                self.step_duration.record(
                    duration,
                    {"step": step_name}
                )

            # Update aggregated metrics
            if step_name not in self.aggregated_metrics["step_durations"]:
                self.aggregated_metrics["step_durations"][step_name] = []
            self.aggregated_metrics["step_durations"][step_name].append(duration)

            # Check threshold
            threshold_key = f"{step_name}_ms"
            if threshold_key in self.thresholds and duration > self.thresholds[threshold_key]:
                logger.warning(
                    f"Step exceeded threshold",
                    step=step_name,
                    duration_ms=duration,
                    threshold_ms=self.thresholds[threshold_key],
                    metadata=metadata
                )

            # End span
            if span:
                span.set_attribute("duration.ms", duration)
                span.end()

    @asynccontextmanager
    async def async_time_step(self, request_id: str, step_name: str, **metadata):
        """Async context manager for timing asynchronous steps"""
        profile = self.active_profiles.get(request_id)
        if not profile:
            yield
            return

        step = StepTiming(
            step_name=step_name,
            start_time=time.time(),
            metadata=metadata
        )
        profile.steps.append(step)

        span = None
        if self.enable_tracing:
            span = self.tracer.start_span(step_name)
            for key, value in metadata.items():
                span.set_attribute(f"step.{key}", str(value))

        try:
            yield step
        finally:
            duration = step.complete()

            # Record step metric
            if self.enable_metrics:
                self.step_duration.record(
                    duration,
                    {"step": step_name}
                )

            # Check threshold
            threshold_key = f"{step_name}_ms"
            if threshold_key in self.thresholds and duration > self.thresholds[threshold_key]:
                logger.warning(
                    f"Async step exceeded threshold",
                    step=step_name,
                    duration_ms=duration,
                    threshold_ms=self.thresholds[threshold_key],
                    metadata=metadata
                )

            # End span
            if span:
                span.set_attribute("duration.ms", duration)
                span.end()

    def record_cache_hit(self, cache_type: str = "semantic"):
        """Record a cache hit"""
        if self.enable_metrics:
            self.cache_hit_counter.add(1, {"cache_type": cache_type})

    def record_cache_miss(self, cache_type: str = "semantic"):
        """Record a cache miss"""
        if self.enable_metrics:
            self.cache_miss_counter.add(1, {"cache_type": cache_type})

    def record_tokens(self, count: int, model: str = "unknown"):
        """Record token usage"""
        if self.enable_metrics:
            self.token_counter.add(count, {"model": model})

    def record_error(self, error_type: str, endpoint: str):
        """Record an error"""
        if self.enable_metrics:
            self.error_counter.add(1, {"error_type": error_type, "endpoint": endpoint})
        self.aggregated_metrics["error_count"] += 1

    def get_request_profile(self, request_id: str) -> Optional[RequestProfile]:
        """Get profile for a specific request"""
        # Check active profiles first
        if request_id in self.active_profiles:
            return self.active_profiles[request_id]

        # Check completed profiles
        for profile in reversed(self.completed_profiles):
            if profile.request_id == request_id:
                return profile

        return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        summary = {
            "active_requests": len(self.active_profiles),
            "completed_requests": len(self.completed_profiles),
            "total_requests": self.aggregated_metrics["request_count"],
            "error_rate": self.aggregated_metrics["error_count"] / max(1, self.aggregated_metrics["request_count"]),
            "step_statistics": {}
        }

        # Calculate step statistics
        for step_name, durations in self.aggregated_metrics["step_durations"].items():
            if durations:
                summary["step_statistics"][step_name] = {
                    "count": len(durations),
                    "avg_ms": statistics.mean(durations),
                    "median_ms": statistics.median(durations),
                    "p95_ms": sorted(durations)[int(0.95 * len(durations))] if len(durations) > 1 else durations[0],
                    "max_ms": max(durations),
                    "min_ms": min(durations)
                }

        # Get recent bottlenecks
        recent_bottlenecks = []
        for profile in self.completed_profiles[-10:]:  # Last 10 requests
            if profile.bottlenecks:
                recent_bottlenecks.extend(profile.bottlenecks[:2])  # Top 2 per request

        # Aggregate bottlenecks by step
        bottleneck_summary = {}
        for bottleneck in recent_bottlenecks:
            step = bottleneck["step"]
            if step not in bottleneck_summary:
                bottleneck_summary[step] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_percentage": 0
                }
            bottleneck_summary[step]["count"] += 1
            bottleneck_summary[step]["total_duration"] += bottleneck["duration_ms"]
            bottleneck_summary[step]["avg_percentage"] += bottleneck["percentage"]

        for step in bottleneck_summary:
            count = bottleneck_summary[step]["count"]
            bottleneck_summary[step]["avg_duration_ms"] = bottleneck_summary[step]["total_duration"] / count
            bottleneck_summary[step]["avg_percentage"] = bottleneck_summary[step]["avg_percentage"] / count

        summary["top_bottlenecks"] = sorted(
            bottleneck_summary.items(),
            key=lambda x: x[1]["avg_duration_ms"],
            reverse=True
        )[:5]

        return summary

    def export_profiles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Export recent request profiles for analysis"""
        profiles = []
        for profile in self.completed_profiles[-limit:]:
            profile_dict = asdict(profile)
            # Add analysis
            profile_dict["analysis"] = {
                "bottleneck_count": len(profile.bottlenecks),
                "slowest_step": profile.bottlenecks[0] if profile.bottlenecks else None,
                "step_count": len(profile.steps)
            }
            profiles.append(profile_dict)
        return profiles

    def get_latency_breakdown(self, request_id: str) -> Dict[str, Any]:
        """Get detailed latency breakdown for a specific request"""
        profile = self.get_request_profile(request_id)
        if not profile:
            return {"error": "Request profile not found"}

        breakdown = {
            "request_id": request_id,
            "total_duration_ms": profile.total_duration_ms,
            "steps": []
        }

        # Calculate step percentages
        for step in profile.steps:
            if step.duration_ms and profile.total_duration_ms:
                percentage = (step.duration_ms / profile.total_duration_ms) * 100
                breakdown["steps"].append({
                    "name": step.step_name,
                    "duration_ms": step.duration_ms,
                    "percentage": round(percentage, 2),
                    "metadata": step.metadata
                })

        # Sort by duration
        breakdown["steps"].sort(key=lambda x: x["duration_ms"], reverse=True)

        # Add waterfall visualization data
        breakdown["waterfall"] = self._generate_waterfall_data(profile)

        return breakdown

    def _generate_waterfall_data(self, profile: RequestProfile) -> List[Dict[str, Any]]:
        """Generate data for waterfall visualization"""
        waterfall = []
        base_time = profile.start_time

        for step in profile.steps:
            waterfall.append({
                "step": step.step_name,
                "start_offset_ms": (step.start_time - base_time) * 1000,
                "duration_ms": step.duration_ms or 0,
                "end_offset_ms": ((step.end_time or step.start_time) - base_time) * 1000
            })

        return waterfall


# Decorator for automatic step timing
def time_function(step_name: Optional[str] = None):
    """Decorator to automatically time function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract request_id from kwargs or first arg if it's a request object
            request_id = kwargs.get("request_id", "unknown")
            if not request_id and args and hasattr(args[0], "request_id"):
                request_id = args[0].request_id

            monitor = kwargs.get("monitor")
            if not monitor:
                # Use the global monitor from this module
                monitor = global_monitor

            if monitor:
                async with monitor.async_time_step(request_id, step_name or func.__name__):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request_id = kwargs.get("request_id", "unknown")
            if not request_id and args and hasattr(args[0], "request_id"):
                request_id = args[0].request_id

            monitor = kwargs.get("monitor")
            if not monitor:
                # Use the global monitor from this module
                monitor = global_monitor

            if monitor:
                with monitor.time_step(request_id, step_name or func.__name__):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Global monitor instance
global_monitor = EnhancedPerformanceMonitor()