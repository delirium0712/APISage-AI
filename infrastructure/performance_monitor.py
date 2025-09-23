"""
Performance Monitor for RAG systems
Real-time metrics tracking and alerting for APISage
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import statistics


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query_id: str
    timestamp: datetime
    query: str
    response_time_ms: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False
    evaluation_scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics."""

    total_queries: int
    success_rate: float
    average_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    total_tokens: int
    cache_hit_rate: float
    error_count: int
    queries_per_minute: float
    time_window_minutes: int


@dataclass
class Alert:
    """Performance alert."""

    alert_id: str
    timestamp: datetime
    level: str  # INFO, WARNING, CRITICAL
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    metadata: Dict[str, Any]


class PerformanceMonitor:
    """
    Real-time performance monitoring for RAG systems.
    Tracks metrics, generates alerts, and provides performance insights.
    """

    def __init__(
        self,
        retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_real_time_alerts: bool = True,
        max_queries_in_memory: int = 10000,
    ):
        """
        Initialize the performance monitor.

        Args:
            retention_hours: How long to keep metrics in memory
            alert_thresholds: Custom alert thresholds
            enable_real_time_alerts: Enable real-time alerting
            max_queries_in_memory: Maximum queries to keep in memory
        """
        self.retention_hours = retention_hours
        self.enable_real_time_alerts = enable_real_time_alerts
        self.max_queries_in_memory = max_queries_in_memory

        # Default alert thresholds
        self.alert_thresholds = {
            "response_time_ms": 5000,  # 5 seconds
            "error_rate": 0.05,  # 5%
            "cache_hit_rate": 0.3,  # 30% minimum
            "queries_per_minute": 100,  # 100 QPM maximum
            "token_usage_per_hour": 100000,  # 100K tokens per hour
        }

        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)

        # Metrics storage
        self.query_metrics: deque = deque(maxlen=self.max_queries_in_memory)
        self.alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts

        # Time-bucketed metrics for efficient aggregation
        self.time_buckets = {
            "1min": defaultdict(list),
            "5min": defaultdict(list),
            "1hour": defaultdict(list),
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_metrics, daemon=True
        )
        self._cleanup_thread.start()

        print(f"Performance monitor initialized with {retention_hours}h retention")

    def record_query(
        self,
        query_id: str,
        query: str,
        response_time_ms: float,
        tokens_used: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        cache_hit: bool = False,
        evaluation_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record metrics for a query.

        Args:
            query_id: Unique identifier for the query
            query: The query text
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens used
            success: Whether the query was successful
            error_message: Error message if failed
            cache_hit: Whether this was a cache hit
            evaluation_scores: RAG evaluation scores
            metadata: Additional metadata
        """
        timestamp = datetime.now()

        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=timestamp,
            query=query,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            success=success,
            error_message=error_message,
            cache_hit=cache_hit,
            evaluation_scores=evaluation_scores or {},
            metadata=metadata or {},
        )

        with self._lock:
            # Store metrics
            self.query_metrics.append(metrics)

            # Update time buckets
            self._update_time_buckets(metrics)

            # Check for alerts
            if self.enable_real_time_alerts:
                self._check_alerts(metrics)

    def _update_time_buckets(self, metrics: QueryMetrics) -> None:
        """Update time-bucketed metrics for efficient aggregation."""
        timestamp = metrics.timestamp

        # 1-minute buckets
        minute_key = timestamp.replace(second=0, microsecond=0)
        self.time_buckets["1min"][minute_key].append(metrics)

        # 5-minute buckets
        five_min_key = timestamp.replace(
            minute=(timestamp.minute // 5) * 5, second=0, microsecond=0
        )
        self.time_buckets["5min"][five_min_key].append(metrics)

        # 1-hour buckets
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        self.time_buckets["1hour"][hour_key].append(metrics)

    def _check_alerts(self, metrics: QueryMetrics) -> None:
        """Check if any alert thresholds are breached."""
        try:
            # Response time alert
            if metrics.response_time_ms > self.alert_thresholds["response_time_ms"]:
                alert = Alert(
                    alert_id=f"response_time_{int(time.time())}",
                    timestamp=datetime.now(),
                    level="WARNING",
                    message=f"High response time: {metrics.response_time_ms:.0f}ms",
                    metric_name="response_time_ms",
                    metric_value=metrics.response_time_ms,
                    threshold=self.alert_thresholds["response_time_ms"],
                    metadata={"query_id": metrics.query_id},
                )
                self._trigger_alert(alert)

            # Error rate alert (check recent queries)
            recent_metrics = self._get_recent_metrics(minutes=5)
            if recent_metrics:
                error_rate = sum(1 for m in recent_metrics if not m.success) / len(
                    recent_metrics
                )
                if error_rate > self.alert_thresholds["error_rate"]:
                    alert = Alert(
                        alert_id=f"error_rate_{int(time.time())}",
                        timestamp=datetime.now(),
                        level="CRITICAL",
                        message=f"High error rate: {error_rate:.1%}",
                        metric_name="error_rate",
                        metric_value=error_rate,
                        threshold=self.alert_thresholds["error_rate"],
                        metadata={"window_minutes": 5},
                    )
                    self._trigger_alert(alert)

            # QPM alert
            recent_count = len(self._get_recent_metrics(minutes=1))
            if recent_count > self.alert_thresholds["queries_per_minute"]:
                alert = Alert(
                    alert_id=f"qpm_{int(time.time())}",
                    timestamp=datetime.now(),
                    level="WARNING",
                    message=f"High query rate: {recent_count} QPM",
                    metric_name="queries_per_minute",
                    metric_value=float(recent_count),
                    threshold=self.alert_thresholds["queries_per_minute"],
                    metadata={"window_minutes": 1},
                )
                self._trigger_alert(alert)

        except Exception as e:
            print(f"Error checking alerts: {e}")

    def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and notify callbacks."""
        with self._lock:
            self.alerts.append(alert)

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Alert callback error: {e}")

    def _get_recent_metrics(self, minutes: int) -> List[QueryMetrics]:
        """Get metrics from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.query_metrics if m.timestamp >= cutoff]

    def get_metrics(self, window_minutes: int = 60) -> AggregatedMetrics:
        """
        Get aggregated metrics for a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            AggregatedMetrics object
        """
        with self._lock:
            recent_metrics = self._get_recent_metrics(window_minutes)

            if not recent_metrics:
                return AggregatedMetrics(
                    total_queries=0,
                    success_rate=0.0,
                    average_response_time_ms=0.0,
                    median_response_time_ms=0.0,
                    p95_response_time_ms=0.0,
                    total_tokens=0,
                    cache_hit_rate=0.0,
                    error_count=0,
                    queries_per_minute=0.0,
                    time_window_minutes=window_minutes,
                )

            # Calculate metrics
            total_queries = len(recent_metrics)
            success_count = sum(1 for m in recent_metrics if m.success)
            success_rate = success_count / total_queries

            response_times = [m.response_time_ms for m in recent_metrics]
            average_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = (
                sorted(response_times)[int(0.95 * len(response_times))]
                if response_times
                else 0.0
            )

            total_tokens = sum(m.tokens_used for m in recent_metrics)
            cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
            cache_hit_rate = cache_hits / total_queries

            error_count = total_queries - success_count
            queries_per_minute = total_queries / window_minutes

            return AggregatedMetrics(
                total_queries=total_queries,
                success_rate=success_rate,
                average_response_time_ms=average_response_time,
                median_response_time_ms=median_response_time,
                p95_response_time_ms=p95_response_time,
                total_tokens=total_tokens,
                cache_hit_rate=cache_hit_rate,
                error_count=error_count,
                queries_per_minute=queries_per_minute,
                time_window_minutes=window_minutes,
            )

    def get_evaluation_metrics(self, window_minutes: int = 60) -> Dict[str, float]:
        """Get aggregated evaluation metrics."""
        with self._lock:
            recent_metrics = self._get_recent_metrics(window_minutes)

            # Collect evaluation scores
            all_scores = defaultdict(list)
            for metrics in recent_metrics:
                if metrics.evaluation_scores:
                    for metric_name, score in metrics.evaluation_scores.items():
                        all_scores[metric_name].append(score)

            # Calculate averages
            avg_scores = {}
            for metric_name, scores in all_scores.items():
                if scores:
                    avg_scores[f"avg_{metric_name}"] = statistics.mean(scores)
                    avg_scores[f"median_{metric_name}"] = statistics.median(scores)
                    avg_scores[f"min_{metric_name}"] = min(scores)
                    avg_scores[f"max_{metric_name}"] = max(scores)

            return avg_scores

    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            return list(self.alerts)[-limit:]

    def get_error_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of recent errors."""
        with self._lock:
            recent_metrics = self._get_recent_metrics(window_minutes)
            failed_metrics = [m for m in recent_metrics if not m.success]

            if not failed_metrics:
                return {"total_errors": 0, "error_types": {}}

            # Group errors by type
            error_types = defaultdict(int)
            for metrics in failed_metrics:
                error_msg = metrics.error_message or "Unknown error"
                # Simple categorization
                if "timeout" in error_msg.lower():
                    error_types["timeout"] += 1
                elif "rate limit" in error_msg.lower():
                    error_types["rate_limit"] += 1
                elif "api" in error_msg.lower():
                    error_types["api_error"] += 1
                else:
                    error_types["other"] += 1

            return {
                "total_errors": len(failed_metrics),
                "error_rate": len(failed_metrics) / len(recent_metrics),
                "error_types": dict(error_types),
                "recent_errors": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "query_id": m.query_id,
                        "error": m.error_message,
                    }
                    for m in failed_metrics[-10:]  # Last 10 errors
                ],
            }

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)

    def clear_metrics(self) -> int:
        """Clear all stored metrics and return count of cleared items."""
        with self._lock:
            count = len(self.query_metrics)
            self.query_metrics.clear()
            self.time_buckets = {
                "1min": defaultdict(list),
                "5min": defaultdict(list),
                "1hour": defaultdict(list),
            }
            return count

    def clear_alerts(self) -> int:
        """Clear all stored alerts and return count of cleared items."""
        with self._lock:
            count = len(self.alerts)
            self.alerts.clear()
            return count

    def _cleanup_old_metrics(self) -> None:
        """Background thread to clean up old metrics."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes

                cutoff = datetime.now() - timedelta(hours=self.retention_hours)

                with self._lock:
                    # Clean up main metrics
                    original_count = len(self.query_metrics)
                    self.query_metrics = deque(
                        (m for m in self.query_metrics if m.timestamp >= cutoff),
                        maxlen=self.max_queries_in_memory,
                    )

                    # Clean up time buckets
                    for bucket_type, buckets in self.time_buckets.items():
                        old_keys = [k for k in buckets.keys() if k < cutoff]
                        for key in old_keys:
                            del buckets[key]

                    # Clean up old alerts (keep alerts for longer)
                    alert_cutoff = datetime.now() - timedelta(
                        hours=self.retention_hours * 2
                    )
                    self.alerts = deque(
                        (a for a in self.alerts if a.timestamp >= alert_cutoff),
                        maxlen=1000,
                    )

                    cleaned_count = original_count - len(self.query_metrics)
                    if cleaned_count > 0:
                        print(f"Cleaned up {cleaned_count} old metrics")

            except Exception as e:
                print(f"Cleanup error: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        metrics_5min = self.get_metrics(window_minutes=5)
        metrics_60min = self.get_metrics(window_minutes=60)
        recent_alerts = self.get_recent_alerts(limit=10)

        # Determine health status
        health_score = 100.0

        # Reduce score based on error rate
        if metrics_5min.success_rate < 0.95:
            health_score -= (1 - metrics_5min.success_rate) * 50

        # Reduce score based on response time
        if metrics_5min.average_response_time_ms > 2000:
            health_score -= min(
                30, (metrics_5min.average_response_time_ms - 2000) / 100
            )

        # Reduce score based on recent critical alerts
        critical_alerts = [a for a in recent_alerts if a.level == "CRITICAL"]
        health_score -= len(critical_alerts) * 10

        health_score = max(0, health_score)

        if health_score >= 90:
            status = "HEALTHY"
        elif health_score >= 70:
            status = "DEGRADED"
        else:
            status = "UNHEALTHY"

        return {
            "status": status,
            "health_score": health_score,
            "metrics_5min": asdict(metrics_5min),
            "metrics_60min": asdict(metrics_60min),
            "recent_alerts_count": len(recent_alerts),
            "critical_alerts_count": len(critical_alerts),
            "timestamp": datetime.now().isoformat(),
        }

    def get_monitor_info(self) -> Dict[str, Any]:
        """Get information about the monitor configuration and status."""
        with self._lock:
            return {
                "retention_hours": self.retention_hours,
                "max_queries_in_memory": self.max_queries_in_memory,
                "enable_real_time_alerts": self.enable_real_time_alerts,
                "alert_thresholds": self.alert_thresholds,
                "current_metrics_count": len(self.query_metrics),
                "current_alerts_count": len(self.alerts),
                "alert_callbacks_count": len(self.alert_callbacks),
                "time_buckets_counts": {
                    bucket: len(data) for bucket, data in self.time_buckets.items()
                },
            }


# Factory function for easy instantiation
def create_performance_monitor(
    retention_hours: int = 24,
    alert_thresholds: Optional[Dict[str, float]] = None,
    enable_real_time_alerts: bool = True,
) -> PerformanceMonitor:
    """
    Factory function to create a configured PerformanceMonitor.

    Args:
        retention_hours: How long to keep metrics
        alert_thresholds: Custom alert thresholds
        enable_real_time_alerts: Enable real-time alerting

    Returns:
        Configured PerformanceMonitor instance
    """
    return PerformanceMonitor(
        retention_hours=retention_hours,
        alert_thresholds=alert_thresholds,
        enable_real_time_alerts=enable_real_time_alerts,
    )


# Global instance for the main API
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get or create the global performance monitor instance.

    Returns:
        Global PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = create_performance_monitor()
    return _performance_monitor
