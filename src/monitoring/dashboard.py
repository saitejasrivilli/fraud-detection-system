"""
MONITORING DASHBOARD
Real-time metrics collection and visualization for production fraud detection
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from collections import deque
import threading
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MonitoringDashboard')


class MetricsCollector:
    """
    Collect and aggregate metrics from all components
    """
    
    def __init__(self, history_size: int = 10000):
        """
        Args:
            history_size: Size of metrics history buffer
        """
        self.history = deque(maxlen=history_size)
        self.lock = threading.Lock()
        self.start_time = datetime.now()
    
    def record_metric(self, metric_type: str, value: float, tags: Dict = None):
        """
        Record a single metric
        
        Args:
            metric_type: Type of metric (e.g., 'latency_ms', 'fraud_rate')
            value: Metric value
            tags: Optional metadata tags
        """
        tags = tags or {}
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': metric_type,
            'value': value,
            'tags': tags
        }
        
        with self.lock:
            self.history.append(entry)
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get summary of metrics from last N minutes
        
        Args:
            minutes: Time window in minutes
        
        Returns:
            Metrics summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent = [
                m for m in self.history
                if datetime.fromisoformat(m['timestamp']) >= cutoff_time
            ]
        
        # Group by type
        metrics_by_type = {}
        for metric in recent:
            mtype = metric['type']
            if mtype not in metrics_by_type:
                metrics_by_type[mtype] = []
            metrics_by_type[mtype].append(metric['value'])
        
        # Calculate stats
        summary = {}
        for mtype, values in metrics_by_type.items():
            if values:
                summary[mtype] = {
                    'count': len(values),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'avg': float(statistics.mean(values)),
                    'p50': float(statistics.median(values)) if len(values) > 1 else float(values[0]),
                    'p95': float(sorted(values)[int(len(values)*0.95)]) if len(values) > 1 else float(values[0])
                }
        
        return summary


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for fraud detection system
    """
    
    def __init__(self):
        """Initialize dashboard"""
        self.metrics_collector = MetricsCollector()
        self.alerts = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Health status
        self.component_status = {
            'isolation_forest': 'healthy',
            'autoencoder': 'healthy',
            'lstm': 'healthy',
            'gcn_batch': 'healthy',
            'manual_review': 'healthy'
        }
        
        # Thresholds
        self.alert_thresholds = {
            'latency_p99_ms': 100,
            'fraud_rate': 0.10,
            'error_rate': 0.05,
            'model_disagreement': 0.30
        }
    
    def record_transaction_score(
        self,
        transaction_id: str,
        latency_ms: float,
        fraud_probability: float,
        model_predictions: Dict[str, bool]
    ):
        """
        Record transaction scoring metrics
        
        Args:
            transaction_id: Transaction ID
            latency_ms: Scoring latency
            fraud_probability: Fraud probability (0-1)
            model_predictions: Predictions from each model
        """
        self.metrics_collector.record_metric(
            'latency_ms',
            latency_ms,
            tags={'transaction_id': transaction_id}
        )
        
        self.metrics_collector.record_metric(
            'fraud_probability',
            fraud_probability,
            tags={'transaction_id': transaction_id}
        )
        
        # Model agreement
        agreements = sum(1 for v in model_predictions.values() if v) / max(len(model_predictions), 1)
        self.metrics_collector.record_metric(
            'model_agreement',
            agreements,
            tags={'transaction_id': transaction_id}
        )
        
        # Check for alerts
        self._check_alerts(latency_ms, fraud_probability, agreements)
    
    def _check_alerts(self, latency_ms: float, fraud_prob: float, agreement: float):
        """Check for alerting conditions"""
        alerts = []
        
        if latency_ms > self.alert_thresholds['latency_p99_ms']:
            alerts.append({
                'severity': 'warning',
                'alert': 'High latency',
                'value': f"{latency_ms:.2f}ms",
                'threshold': f"{self.alert_thresholds['latency_p99_ms']}ms",
                'timestamp': datetime.now().isoformat()
            })
        
        if fraud_prob > 0.95:
            alerts.append({
                'severity': 'critical',
                'alert': 'Very high fraud score',
                'value': f"{fraud_prob:.2f}",
                'timestamp': datetime.now().isoformat()
            })
        
        if agreement < (1 - self.alert_thresholds['model_disagreement']):
            alerts.append({
                'severity': 'info',
                'alert': 'Model disagreement detected',
                'value': f"{agreement:.2f}",
                'timestamp': datetime.now().isoformat()
            })
        
        for alert in alerts:
            with self.lock:
                self.alerts.append(alert)
    
    def record_batch_job(self, job_name: str, duration_seconds: float, status: str):
        """
        Record batch job execution
        
        Args:
            job_name: Batch job name (e.g., 'gcn_analysis')
            duration_seconds: Execution time
            status: 'success' or 'failed'
        """
        self.metrics_collector.record_metric(
            f'batch_duration_{job_name}',
            duration_seconds,
            tags={'status': status}
        )
        
        if status == 'failed':
            with self.lock:
                self.alerts.append({
                    'severity': 'critical',
                    'alert': f'Batch job failed: {job_name}',
                    'timestamp': datetime.now().isoformat()
                })
    
    def update_component_status(self, component: str, status: str):
        """
        Update component health status
        
        Args:
            component: Component name
            status: 'healthy', 'degraded', or 'failed'
        """
        self.component_status[component] = status
        
        if status != 'healthy':
            with self.lock:
                self.alerts.append({
                    'severity': 'critical' if status == 'failed' else 'warning',
                    'alert': f'Component status: {component} - {status}',
                    'timestamp': datetime.now().isoformat()
                })
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data
        
        Returns:
            Dashboard JSON
        """
        metrics_summary = self.metrics_collector.get_metrics_summary(minutes=60)
        
        with self.lock:
            recent_alerts = list(self.alerts)[-20:]  # Last 20 alerts
        
        # Calculate statistics
        latency_stats = metrics_summary.get('latency_ms', {})
        fraud_rate = (
            metrics_summary.get('fraud_probability', {}).get('avg', 0)
        )
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'system_health': {
                'overall_status': self._calculate_overall_status(),
                'component_status': self.component_status
            },
            'performance': {
                'latency': latency_stats,
                'fraud_rate': float(fraud_rate),
                'model_agreement': metrics_summary.get('model_agreement', {})
            },
            'sla_compliance': {
                'latency_p99_sla': 100,  # ms
                'latency_p99_actual': latency_stats.get('p95', 0),
                'sla_met': latency_stats.get('p95', 0) <= 100
            },
            'recent_alerts': recent_alerts,
            'metrics_60min': metrics_summary
        }
        
        return dashboard
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system health"""
        failed = sum(1 for s in self.component_status.values() if s == 'failed')
        degraded = sum(1 for s in self.component_status.values() if s == 'degraded')
        
        if failed > 0:
            return 'critical'
        elif degraded > 1:
            return 'degraded'
        else:
            return 'healthy'
    
    def export_dashboard_html(self, output_path: str = "dashboard.html"):
        """
        Export dashboard as HTML
        
        Args:
            output_path: Path to save HTML file
        """
        dashboard_data = self.get_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection - Monitoring Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: #1a1a1a; color: white; padding: 20px; border-radius: 5px; }}
                .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric {{ font-size: 24px; font-weight: bold; color: #1a1a1a; }}
                .label {{ color: #666; margin-top: 10px; }}
                .status-healthy {{ color: #4CAF50; }}
                .status-degraded {{ color: #FF9800; }}
                .status-critical {{ color: #F44336; }}
                .alerts {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ Fraud Detection System - Monitoring Dashboard</h1>
                    <p>Last updated: {dashboard_data['timestamp']}</p>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="metric status-{dashboard_data['system_health']['overall_status']}">
                            {dashboard_data['system_health']['overall_status'].upper()}
                        </div>
                        <div class="label">System Status</div>
                    </div>
                    
                    <div class="card">
                        <div class="metric">{dashboard_data['performance']['latency'].get('avg', 0):.1f}ms</div>
                        <div class="label">Avg Latency</div>
                    </div>
                    
                    <div class="card">
                        <div class="metric">{dashboard_data['performance']['fraud_rate']:.2%}</div>
                        <div class="label">Fraud Rate (60min)</div>
                    </div>
                </div>
                
                <h2>Component Status</h2>
                <div class="card">
                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Status</th>
                        </tr>
        """
        
        for component, status in dashboard_data['system_health']['component_status'].items():
            status_class = f"status-{status}"
            html += f"""
                        <tr>
                            <td>{component}</td>
                            <td><span class="{status_class}">{status.upper()}</span></td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <h2>Recent Alerts</h2>
                <div class="alerts">
        """
        
        for alert in dashboard_data['recent_alerts'][-10:]:
            severity_class = f"status-{alert['severity']}"
            html += f"""
                    <div style="padding: 10px; border-left: 4px solid; margin: 10px 0;" class="{severity_class}">
                        <strong>{alert['alert']}</strong> - {alert.get('value', '')}
                        <br><small>{alert['timestamp']}</small>
                    </div>
            """
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Dashboard exported to {output_path}")
    
    def export_metrics_json(self, output_path: str = "metrics.json"):
        """Export metrics as JSON"""
        dashboard_data = self.get_dashboard_data()
        
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {output_path}")


# Prometheus-style metrics exporter
class PrometheusExporter:
    """
    Export metrics in Prometheus format for integration with monitoring stack
    """
    
    def __init__(self, dashboard: MonitoringDashboard):
        """
        Args:
            dashboard: MonitoringDashboard instance
        """
        self.dashboard = dashboard
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics"""
        dashboard_data = self.dashboard.get_dashboard_data()
        
        metrics = []
        metrics.append("# HELP fraud_detection_system_metrics Fraud Detection System Metrics")
        metrics.append("# TYPE fraud_detection_system_metrics gauge")
        
        # Latency metrics
        latency = dashboard_data['performance']['latency']
        metrics.append(f"fraud_detection_latency_p50_ms {latency.get('p50', 0)}")
        metrics.append(f"fraud_detection_latency_p95_ms {latency.get('p95', 0)}")
        metrics.append(f"fraud_detection_latency_avg_ms {latency.get('avg', 0)}")
        
        # Fraud rate
        metrics.append(
            f"fraud_detection_fraud_rate "
            f"{dashboard_data['performance']['fraud_rate']}"
        )
        
        # Component status
        for component, status in dashboard_data['system_health']['component_status'].items():
            status_code = {'healthy': 1, 'degraded': 0.5, 'failed': 0}.get(status, 0)
            metrics.append(f'fraud_detection_component_health{{component="{component}"}} {status_code}')
        
        # Alert count
        metrics.append(f"fraud_detection_active_alerts {len(dashboard_data['recent_alerts'])}")
        
        return "\n".join(metrics)


if __name__ == "__main__":
    # Example usage
    dashboard = MonitoringDashboard()
    
    # Simulate some transactions
    dashboard.record_transaction_score(
        transaction_id="TXN_001",
        latency_ms=45.2,
        fraud_probability=0.92,
        model_predictions={
            'if': True,
            'ae': True,
            'lstm': True,
            'gcn': False
        }
    )
    
    # Get dashboard data
    data = dashboard.get_dashboard_data()
    print("\n📊 DASHBOARD DATA:")
    print(json.dumps(data, indent=2, default=str))
    
    # Export
    dashboard.export_dashboard_html()
    dashboard.export_metrics_json()
    
    print("\n✓ Dashboard exported to dashboard.html and metrics.json")
