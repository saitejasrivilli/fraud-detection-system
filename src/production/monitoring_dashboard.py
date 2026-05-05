"""
Monitoring Dashboard
Real-time metrics, alerts, and system health tracking
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, deque
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self, window_size_hours: int = 24):
        """
        Args:
            window_size_hours: Time window for metrics (default 24h)
        """
        self.window_size = timedelta(hours=window_size_hours)
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.start_time = datetime.now()
    
    def record_metric(self, name: str, value: float, 
                     timestamp: datetime = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        values = [m['value'] for m in self.metrics[name]]
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99))
        }


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for fraud detection system
    Tracks performance, alerts, and system health
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = deque(maxlen=1000)
        self.sla_violations = []
        self.health_checks = deque(maxlen=100)
        
        # SLA thresholds
        self.sla_thresholds = {
            'latency_ms': 50,           # p99 latency < 50ms
            'error_rate': 1.0,          # < 1% error
            'fraud_detection_rate': 0.5,# >= 50% of fraud detected
            'precision': 85.0            # >= 85% precision
        }
        
        # Key metrics
        self.current_metrics = {
            'total_transactions': 0,
            'transactions_flagged': 0,
            'total_errors': 0,
            'model_status': {},
            'api_health': 'HEALTHY'
        }
    
    def record_transaction(self, prediction: Dict[str, Any]):
        """
        Record transaction prediction
        """
        self.metrics.record_metric('total_transactions', 1.0)
        self.current_metrics['total_transactions'] += 1
        
        if prediction.get('is_fraud'):
            self.metrics.record_metric('transactions_flagged', 1.0)
            self.current_metrics['transactions_flagged'] += 1
        
        if 'error' in prediction:
            self.metrics.record_metric('total_errors', 1.0)
            self.current_metrics['total_errors'] += 1
        
        # Record latency
        latency = prediction.get('latency_ms', 0)
        self.metrics.record_metric('latency_ms', latency)
        
        # Check SLA violations
        if latency > self.sla_thresholds['latency_ms']:
            self._create_alert(
                'SLA_VIOLATION',
                'HIGH',
                f"Latency {latency:.2f}ms exceeds SLA ({self.sla_thresholds['latency_ms']}ms)"
            )
    
    def record_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """
        Record model performance metrics
        """
        self.current_metrics['model_status'][model_name] = {
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'auc_roc': metrics.get('auc_roc', 0),
            'last_updated': datetime.now().isoformat()
        }
        
        # Record individual metrics
        for metric_name, value in metrics.items():
            self.metrics.record_metric(f'{model_name}_{metric_name}', value)
    
    def _create_alert(self, alert_type: str, severity: str, message: str):
        """Create an alert"""
        alert = {
            'type': alert_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        
        log_level = logging.WARNING if severity == 'HIGH' else logging.INFO
        logger.log(log_level, f"[{alert_type}] {message}")
    
    def check_sla_compliance(self) -> Dict[str, Any]:
        """Check if system meets SLAs"""
        compliance = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'checks': {}
        }
        
        # Latency SLA
        latency_stats = self.metrics.get_metric_stats('latency_ms')
        if latency_stats:
            p99_latency = latency_stats.get('p99', 0)
            latency_ok = p99_latency < self.sla_thresholds['latency_ms']
            compliance['checks']['latency_p99'] = {
                'status': 'PASS' if latency_ok else 'FAIL',
                'value_ms': p99_latency,
                'threshold_ms': self.sla_thresholds['latency_ms']
            }
            if not latency_ok:
                compliance['overall_status'] = 'DEGRADED'
        
        # Error rate SLA
        if self.current_metrics['total_transactions'] > 0:
            error_rate = (self.current_metrics['total_errors'] / 
                         self.current_metrics['total_transactions'] * 100)
            error_ok = error_rate < self.sla_thresholds['error_rate']
            compliance['checks']['error_rate'] = {
                'status': 'PASS' if error_ok else 'FAIL',
                'value_percent': error_rate,
                'threshold_percent': self.sla_thresholds['error_rate']
            }
            if not error_ok:
                compliance['overall_status'] = 'DEGRADED'
        
        # Model precision
        for model_name, model_metrics in self.current_metrics['model_status'].items():
            precision = model_metrics.get('precision', 0) * 100
            precision_ok = precision >= self.sla_thresholds['precision']
            compliance['checks'][f'{model_name}_precision'] = {
                'status': 'PASS' if precision_ok else 'FAIL',
                'value_percent': precision,
                'threshold_percent': self.sla_thresholds['precision']
            }
            if not precision_ok:
                compliance['overall_status'] = 'DEGRADED'
        
        return compliance
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard display"""
        sla_check = self.check_sla_compliance()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_uptime_hours': (
                (datetime.now() - self.metrics.start_time).total_seconds() / 3600
            ),
            'sla_compliance': sla_check['overall_status'],
            'metrics': {
                'total_transactions': self.current_metrics['total_transactions'],
                'transactions_flagged': self.current_metrics['transactions_flagged'],
                'fraud_rate_percent': (
                    self.current_metrics['transactions_flagged'] / 
                    max(self.current_metrics['total_transactions'], 1) * 100
                ),
                'error_count': self.current_metrics['total_errors'],
                'error_rate_percent': (
                    self.current_metrics['total_errors'] / 
                    max(self.current_metrics['total_transactions'], 1) * 100
                )
            },
            'latency': self.metrics.get_metric_stats('latency_ms'),
            'model_status': self.current_metrics['model_status'],
            'recent_alerts': list(self.alerts)[-10:],  # Last 10
            'sla_details': sla_check['checks']
        }
        
        return summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_dashboard_summary(),
            'all_alerts': list(self.alerts),
            'model_details': {
                model_name: {
                    'status': status,
                    'metrics': self.metrics.get_metric_stats(
                        f"{model_name}_auc_roc"
                    )
                }
                for model_name, status in self.current_metrics['model_status'].items()
            }
        }
    
    def export_dashboard(self, filepath: str):
        """Export dashboard to JSON"""
        try:
            dashboard_data = {
                'dashboard': self.get_dashboard_summary(),
                'report': self.get_performance_report(),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"✓ Dashboard exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
    
    def print_dashboard(self):
        """Print dashboard to console"""
        dashboard = self.get_dashboard_summary()
        
        print("\n" + "="*100)
        print("FRAUD DETECTION MONITORING DASHBOARD")
        print("="*100)
        print(f"\nTimestamp: {dashboard['timestamp']}")
        print(f"SLA Compliance: {dashboard['sla_compliance']}")
        print(f"System Uptime: {dashboard['system_uptime_hours']:.2f} hours")
        
        print("\n" + "-"*100)
        print("TRANSACTION METRICS:")
        print("-"*100)
        metrics = dashboard['metrics']
        print(f"  Total Transactions: {metrics['total_transactions']:,}")
        print(f"  Flagged as Fraud: {metrics['transactions_flagged']:,} "
              f"({metrics['fraud_rate_percent']:.2f}%)")
        print(f"  Errors: {metrics['error_count']} "
              f"({metrics['error_rate_percent']:.2f}%)")
        
        print("\n" + "-"*100)
        print("LATENCY METRICS:")
        print("-"*100)
        latency = dashboard['latency']
        if latency:
            print(f"  Mean: {latency['mean']:.2f}ms")
            print(f"  P95: {latency['p95']:.2f}ms")
            print(f"  P99: {latency['p99']:.2f}ms")
        
        print("\n" + "-"*100)
        print("MODEL STATUS:")
        print("-"*100)
        for model_name, status in dashboard['model_status'].items():
            print(f"  {model_name:20} | Precision: {status['precision']*100:5.1f}% | "
                  f"Recall: {status['recall']*100:5.1f}%")
        
        print("\n" + "-"*100)
        print("SLA COMPLIANCE:")
        print("-"*100)
        for check_name, check_result in dashboard['sla_details'].items():
            status = "✓ PASS" if check_result['status'] == 'PASS' else "✗ FAIL"
            value = check_result.get('value_ms') or check_result.get('value_percent', 0)
            threshold = check_result.get('threshold_ms') or check_result.get('threshold_percent', 0)
            print(f"  {check_name:30} {status:8} ({value:.2f} / {threshold:.2f})")
        
        if dashboard['recent_alerts']:
            print("\n" + "-"*100)
            print("RECENT ALERTS:")
            print("-"*100)
            for alert in dashboard['recent_alerts']:
                print(f"  [{alert['severity']:6}] {alert['type']:20} - {alert['message']}")
        
        print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    # Example usage
    dashboard = MonitoringDashboard()
    
    # Simulate transactions
    for i in range(100):
        prediction = {
            'is_fraud': np.random.random() < 0.01,
            'latency_ms': np.random.exponential(10) + 5,
            'error': None if np.random.random() > 0.005 else "Sample error"
        }
        dashboard.record_transaction(prediction)
    
    # Record model metrics
    dashboard.record_model_metrics('Isolation Forest', {
        'precision': 0.91,
        'recall': 0.72,
        'f1': 0.80,
        'auc_roc': 0.847
    })
    
    dashboard.record_model_metrics('GCN', {
        'precision': 0.87,
        'recall': 0.78,
        'f1': 0.82,
        'auc_roc': 0.859
    })
    
    # Print dashboard
    dashboard.print_dashboard()
    
    # Export
    dashboard.export_dashboard('monitoring_dashboard.json')
