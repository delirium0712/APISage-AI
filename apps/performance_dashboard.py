"""
Performance Dashboard for APISage
Real-time monitoring, bottleneck analysis, and optimization recommendations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

import httpx
import structlog

logger = structlog.get_logger()

# Dashboard configuration
API_BASE_URL = "http://localhost:8080"
REFRESH_INTERVAL = 5  # seconds


class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        self.performance_data = []
        self.cache_data = []
        self.error_data = []
        self.bottleneck_data = []

        # Background data collection
        self.is_monitoring = False
        self.monitor_thread = None

    async def fetch_metrics(self) -> Dict[str, Any]:
        """Fetch latest performance metrics from API"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/metrics")
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
            return {}

    async def fetch_health(self) -> Dict[str, Any]:
        """Fetch health status"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to fetch health: {e}")
            return {}

    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Fetch data asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                metrics = loop.run_until_complete(self.fetch_metrics())
                health = loop.run_until_complete(self.fetch_health())

                # Store data
                timestamp = datetime.now()

                if metrics:
                    # Performance data
                    perf_data = metrics.get("performance", {})
                    perf_data["timestamp"] = timestamp
                    self.performance_data.append(perf_data)

                    # Cache data
                    cache_data = metrics.get("cache", {})
                    cache_data["timestamp"] = timestamp
                    self.cache_data.append(cache_data)

                    # Bottleneck data
                    profiles = metrics.get("recent_profiles", [])
                    for profile in profiles:
                        if profile.get("bottlenecks"):
                            for bottleneck in profile["bottlenecks"]:
                                bottleneck["timestamp"] = timestamp
                                bottleneck["request_id"] = profile["request_id"]
                                self.bottleneck_data.append(bottleneck)

                # Keep only recent data (last hour)
                cutoff = timestamp - timedelta(hours=1)
                self.performance_data = [
                    d for d in self.performance_data[-1000:]
                    if d.get("timestamp", timestamp) > cutoff
                ]
                self.cache_data = [
                    d for d in self.cache_data[-1000:]
                    if d.get("timestamp", timestamp) > cutoff
                ]
                self.bottleneck_data = [
                    d for d in self.bottleneck_data[-2000:]
                    if d.get("timestamp", timestamp) > cutoff
                ]

                loop.close()

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(REFRESH_INTERVAL)

    def create_performance_chart(self) -> go.Figure:
        """Create performance metrics chart"""
        if not self.performance_data:
            return go.Figure().add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        df = pd.DataFrame(self.performance_data)

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Request Count", "Error Rate",
                "Response Time Distribution", "Active Requests",
                "Step Statistics (Avg Duration)", "Recent Bottlenecks"
            ],
            specs=[
                [{"secondary_y": True}, {"type": "indicator"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )

        if 'completed_requests' in df.columns:
            # Request count over time
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['completed_requests'],
                    mode='lines+markers',
                    name='Completed Requests',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

        if 'error_rate' in df.columns:
            # Error rate indicator
            latest_error_rate = df['error_rate'].iloc[-1] if len(df) > 0 else 0
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=latest_error_rate * 100,
                    title={'text': "Error Rate (%)"},
                    gauge={'axis': {'range': [0, 10]},
                           'bar': {'color': "red" if latest_error_rate > 0.05 else "green"},
                           'threshold': {'line': {'color': "orange", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 5}}
                ),
                row=1, col=2
            )

        # Step statistics
        if 'step_statistics' in df.columns and df['step_statistics'].notna().any():
            latest_stats = df['step_statistics'].iloc[-1] if len(df) > 0 else {}
            if isinstance(latest_stats, dict):
                steps = list(latest_stats.keys())
                avg_durations = [latest_stats[step].get('avg_ms', 0) for step in steps]

                fig.add_trace(
                    go.Bar(
                        x=steps,
                        y=avg_durations,
                        name='Avg Duration (ms)',
                        marker_color='lightblue'
                    ),
                    row=2, col=2
                )

        fig.update_layout(height=800, showlegend=True, title_text="Performance Dashboard")
        return fig

    def create_cache_chart(self) -> go.Figure:
        """Create cache performance chart"""
        if not self.cache_data:
            return go.Figure().add_annotation(
                text="No cache data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        df = pd.DataFrame(self.cache_data)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Cache Hit Rate", "L1 vs L2 Cache Size",
                "Cache Hit/Miss Trends", "Compression Statistics"
            ]
        )

        # Cache hit rate
        if 'hit_rate' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['hit_rate'] * 100,
                    mode='lines+markers',
                    name='Hit Rate (%)',
                    line=dict(color='green')
                ),
                row=1, col=1
            )

        # Cache sizes
        if 'l1_size' in df.columns and 'l2_size' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['l1_size'],
                    mode='lines',
                    name='L1 Cache Size',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['l2_size'],
                    mode='lines',
                    name='L2 Cache Size',
                    line=dict(color='orange')
                ),
                row=1, col=2
            )

        # Hit/miss trends
        if 'l1_hits' in df.columns and 'l1_misses' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['l1_hits'],
                    mode='lines',
                    name='L1 Hits',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['l1_misses'],
                    mode='lines',
                    name='L1 Misses',
                    line=dict(color='red')
                ),
                row=2, col=1
            )

        # Compression stats
        if 'compressions' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['compressions'],
                    name='Compressions',
                    marker_color='purple'
                ),
                row=2, col=2
            )

        fig.update_layout(height=600, showlegend=True, title_text="Cache Performance")
        return fig

    def create_bottleneck_chart(self) -> go.Figure:
        """Create bottleneck analysis chart"""
        if not self.bottleneck_data:
            return go.Figure().add_annotation(
                text="No bottleneck data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        df = pd.DataFrame(self.bottleneck_data)

        # Group by step and calculate statistics
        step_stats = df.groupby('step').agg({
            'duration_ms': ['count', 'mean', 'max'],
            'percentage': 'mean'
        }).round(2)

        step_stats.columns = ['count', 'avg_duration', 'max_duration', 'avg_percentage']
        step_stats = step_stats.reset_index()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Top Bottlenecks by Frequency", "Top Bottlenecks by Duration",
                "Bottleneck Timeline", "Bottleneck Percentage Impact"
            ]
        )

        # Top bottlenecks by frequency
        fig.add_trace(
            go.Bar(
                x=step_stats['step'],
                y=step_stats['count'],
                name='Frequency',
                marker_color='orange'
            ),
            row=1, col=1
        )

        # Top bottlenecks by duration
        fig.add_trace(
            go.Bar(
                x=step_stats['step'],
                y=step_stats['avg_duration'],
                name='Avg Duration (ms)',
                marker_color='red'
            ),
            row=1, col=2
        )

        # Timeline
        if 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['duration_ms'],
                    mode='markers',
                    marker=dict(
                        color=df['percentage'],
                        size=df['duration_ms'] / 50,  # Scale marker size
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Percentage")
                    ),
                    text=df['step'],
                    name='Bottlenecks'
                ),
                row=2, col=1
            )

        # Percentage impact
        fig.add_trace(
            go.Bar(
                x=step_stats['step'],
                y=step_stats['avg_percentage'],
                name='Avg % Impact',
                marker_color='darkred'
            ),
            row=2, col=2
        )

        fig.update_layout(height=700, showlegend=True, title_text="Bottleneck Analysis")
        return fig

    def generate_optimization_recommendations(self) -> str:
        """Generate optimization recommendations based on current data"""
        recommendations = []

        # Analyze cache performance
        if self.cache_data:
            latest_cache = self.cache_data[-1] if self.cache_data else {}
            hit_rate = latest_cache.get('hit_rate', 0)

            if hit_rate < 0.3:
                recommendations.append(
                    "🔴 **Low Cache Hit Rate**: Consider increasing cache sizes or improving cache key strategies. "
                    f"Current hit rate: {hit_rate:.1%}"
                )
            elif hit_rate > 0.8:
                recommendations.append(
                    f"✅ **Excellent Cache Performance**: Hit rate is {hit_rate:.1%}. Consider expanding cache to other operations."
                )

        # Analyze bottlenecks
        if self.bottleneck_data:
            df = pd.DataFrame(self.bottleneck_data[-100:])  # Last 100 bottlenecks

            if not df.empty:
                top_bottleneck = df.groupby('step')['duration_ms'].mean().idxmax()
                avg_duration = df.groupby('step')['duration_ms'].mean().max()

                recommendations.append(
                    f"🔴 **Primary Bottleneck**: `{top_bottleneck}` averaging {avg_duration:.0f}ms. "
                    "Focus optimization efforts here."
                )

                # Specific recommendations by step type
                if 'llm_generation' in top_bottleneck:
                    recommendations.append(
                        "💡 **LLM Optimization**: Consider request batching, model switching, or prompt optimization."
                    )
                elif 'cache_lookup' in top_bottleneck:
                    recommendations.append(
                        "💡 **Cache Optimization**: Consider cache warming or indexing improvements."
                    )
                elif 'context_retrieval' in top_bottleneck:
                    recommendations.append(
                        "💡 **Retrieval Optimization**: Consider parallel retrieval or index optimization."
                    )

        # Analyze performance trends
        if self.performance_data:
            latest_perf = self.performance_data[-1] if self.performance_data else {}
            error_rate = latest_perf.get('error_rate', 0)

            if error_rate > 0.05:
                recommendations.append(
                    f"🔴 **High Error Rate**: {error_rate:.1%} of requests are failing. Check logs for common error patterns."
                )

        # General recommendations
        recommendations.extend([
            "💡 **Monitoring**: Set up alerts for response time > 2000ms and error rate > 5%",
            "💡 **Scaling**: Consider horizontal scaling if active requests consistently > 50",
            "💡 **Optimization**: Implement request deduplication for identical queries",
            "💡 **Caching**: Use semantic caching for similar questions with different wording"
        ])

        if not recommendations:
            recommendations.append("✅ **System Healthy**: No critical issues detected. Continue monitoring.")

        return "\n\n".join(recommendations)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "monitoring_active": self.is_monitoring,
            "data_points": {
                "performance": len(self.performance_data),
                "cache": len(self.cache_data),
                "bottlenecks": len(self.bottleneck_data)
            }
        }

        # Add latest metrics
        if self.performance_data:
            latest = self.performance_data[-1]
            stats["latest_performance"] = {
                "completed_requests": latest.get("completed_requests", 0),
                "active_requests": latest.get("active_requests", 0),
                "error_rate": f"{latest.get('error_rate', 0):.1%}",
                "average_response_time": f"{latest.get('average_response_time', 0):.0f}ms"
            }

        if self.cache_data:
            latest = self.cache_data[-1]
            stats["latest_cache"] = {
                "hit_rate": f"{latest.get('hit_rate', 0):.1%}",
                "l1_size": latest.get("l1_size", 0),
                "l2_size": latest.get("l2_size", 0),
                "total_hits": latest.get("l1_hits", 0) + latest.get("l2_hits", 0)
            }

        return stats


# Initialize dashboard
dashboard = PerformanceDashboard()


def create_dashboard_interface():
    """Create Gradio interface for performance dashboard"""

    def start_monitoring():
        dashboard.start_monitoring()
        return "✅ Monitoring started"

    def stop_monitoring():
        dashboard.stop_monitoring()
        return "⏹️ Monitoring stopped"

    def refresh_charts():
        perf_chart = dashboard.create_performance_chart()
        cache_chart = dashboard.create_cache_chart()
        bottleneck_chart = dashboard.create_bottleneck_chart()
        return perf_chart, cache_chart, bottleneck_chart

    def get_recommendations():
        return dashboard.generate_optimization_recommendations()

    def get_stats():
        return json.dumps(dashboard.get_current_stats(), indent=2)

    with gr.Blocks(
        title="APISage Performance Dashboard",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    ) as interface:

        gr.Markdown("""
        # 📊 APISage Performance Dashboard

        Real-time monitoring, bottleneck analysis, and optimization recommendations
        """)

        with gr.Row():
            with gr.Column(scale=2):
                start_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Monitoring", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh Charts", variant="secondary")

            with gr.Column(scale=3):
                status_display = gr.Textbox(
                    label="Status",
                    value="Click 'Start Monitoring' to begin",
                    interactive=False
                )

        with gr.Tabs():
            with gr.TabItem("📈 Performance Metrics"):
                performance_plot = gr.Plot(label="Performance Overview")

            with gr.TabItem("🚀 Cache Analysis"):
                cache_plot = gr.Plot(label="Cache Performance")

            with gr.TabItem("🔍 Bottleneck Analysis"):
                bottleneck_plot = gr.Plot(label="Bottleneck Analysis")

            with gr.TabItem("💡 Recommendations"):
                recommendations_text = gr.Markdown(
                    "Click 'Refresh Charts' to generate recommendations"
                )

            with gr.TabItem("📊 Current Stats"):
                stats_text = gr.Textbox(
                    label="Current Statistics (JSON)",
                    lines=20,
                    max_lines=30
                )

        # Event handlers
        start_btn.click(
            fn=start_monitoring,
            outputs=status_display
        )

        stop_btn.click(
            fn=stop_monitoring,
            outputs=status_display
        )

        refresh_btn.click(
            fn=refresh_charts,
            outputs=[performance_plot, cache_plot, bottleneck_plot]
        )

        refresh_btn.click(
            fn=get_recommendations,
            outputs=recommendations_text
        )

        refresh_btn.click(
            fn=get_stats,
            outputs=stats_text
        )

        # Note: Auto-refresh removed for compatibility with Gradio 3.50.2
        # For auto-refresh, upgrade to Gradio 4.x or manually refresh using the button

    return interface


if __name__ == "__main__":
    interface = create_dashboard_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )