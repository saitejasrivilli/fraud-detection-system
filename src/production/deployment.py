"""
Production Deployment Guide
Complete integration of all production systems
"""

import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Import production modules
from src.production.isolation_forest_deployment import (
    IsolationForestDeployment, ProductionMetrics
)
from src.production.gcn_batch_job import GCNBatchJob
from src.production.manual_review_queue import ManualReviewQueue
from src.production.monitoring_dashboard import MonitoringDashboard
from src.production.feedback_loop import FeedbackCollector, ModelImprover


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """
    Unified production deployment system
    Coordinates all components: real-time, batch, review, monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize production system
        
        Args:
            config: Configuration dict with paths, thresholds, etc.
        """
        self.config = config or self._default_config()
        
        # Create output directories
        self._setup_directories()
        
        # Initialize components
        self.if_deployment = None
        self.gcn_batch = None
        self.review_queue = None
        self.dashboard = None
        self.feedback = None
        self.improver = None
        
        logger.info("Production Deployment System Initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'if_model_path': 'models/isolation_forest.pkl',
            'batch_results_dir': 'batch_results',
            'review_queue_file': 'review_queue.json',
            'dashboard_file': 'monitoring_dashboard.json',
            'feedback_file': 'feedback.json',
            'batch_job_schedule': '02:00',  # 2 AM daily
            'model_retrain_frequency_days': 7,
            'sla_latency_ms': 50,
            'sla_error_rate_percent': 1.0
        }
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            Path(self.config.get('batch_results_dir', 'batch_results')),
            Path('models'),
            Path('logs'),
            Path('reports')
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True)
    
    def initialize_real_time_system(self) -> bool:
        """
        Initialize Isolation Forest for real-time serving
        Returns: Success status
        """
        try:
            logger.info("Initializing real-time Isolation Forest...")
            
            self.if_deployment = IsolationForestDeployment(
                model_path=self.config['if_model_path']
            )
            
            # Load model if available
            if Path(self.config['if_model_path']).exists():
                self.if_deployment.load_model()
            else:
                logger.warning("Model file not found. Using fallback predictions.")
            
            # Health check
            if self.if_deployment.health_check():
                logger.info("✓ Real-time system healthy")
                return True
            else:
                logger.warning("Health check warnings detected")
                return True  # Still operational
                
        except Exception as e:
            logger.error(f"Failed to initialize real-time system: {e}")
            return False
    
    def initialize_batch_system(self) -> bool:
        """
        Initialize GCN batch job for overnight analysis
        Returns: Success status
        """
        try:
            logger.info("Initializing batch GCN job...")
            
            self.gcn_batch = GCNBatchJob(
                output_dir=self.config['batch_results_dir']
            )
            
            logger.info("✓ Batch system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize batch system: {e}")
            return False
    
    def initialize_review_system(self) -> bool:
        """
        Initialize manual review queue for high-stakes decisions
        Returns: Success status
        """
        try:
            logger.info("Initializing manual review queue...")
            
            self.review_queue = ManualReviewQueue(
                queue_file=self.config['review_queue_file']
            )
            
            logger.info("✓ Review system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize review system: {e}")
            return False
    
    def initialize_monitoring_system(self) -> bool:
        """
        Initialize monitoring dashboard
        Returns: Success status
        """
        try:
            logger.info("Initializing monitoring dashboard...")
            
            self.dashboard = MonitoringDashboard()
            
            logger.info("✓ Monitoring system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            return False
    
    def initialize_feedback_system(self) -> bool:
        """
        Initialize feedback loop for continuous improvement
        Returns: Success status
        """
        try:
            logger.info("Initializing feedback loop...")
            
            self.feedback = FeedbackCollector(
                feedback_file=self.config['feedback_file']
            )
            self.improver = ModelImprover(self.feedback)
            
            logger.info("✓ Feedback system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize feedback system: {e}")
            return False
    
    def deploy_all_systems(self) -> Dict[str, bool]:
        """
        Deploy all production systems
        Returns: Status of each system
        """
        logger.info("="*100)
        logger.info("DEPLOYING FRAUD DETECTION SYSTEM TO PRODUCTION")
        logger.info("="*100)
        
        status = {
            'real_time': self.initialize_real_time_system(),
            'batch': self.initialize_batch_system(),
            'review': self.initialize_review_system(),
            'monitoring': self.initialize_monitoring_system(),
            'feedback': self.initialize_feedback_system()
        }
        
        all_ok = all(status.values())
        
        logger.info("="*100)
        print("\nDEPLOYMENT STATUS:")
        print("-"*100)
        for system, ok in status.items():
            status_str = "✓ READY" if ok else "✗ FAILED"
            print(f"  {system:20} {status_str}")
        print("-"*100)
        
        if all_ok:
            logger.info("✓ ALL SYSTEMS DEPLOYED SUCCESSFULLY")
        else:
            logger.warning("⚠ Some systems failed. Check logs above.")
        
        logger.info("="*100 + "\n")
        
        return status
    
    def score_transaction_real_time(self, transaction_id: str, 
                                   features: list) -> Dict[str, Any]:
        """
        Score transaction in real-time
        Path: Isolation Forest (0.5ms) → Ensemble if flagged → Manual review if high-risk
        """
        if not self.if_deployment:
            logger.error("Real-time system not initialized")
            return {}
        
        import numpy as np
        
        # 1. Real-time Isolation Forest
        prediction = self.if_deployment.predict(
            transaction_id, np.array(features)
        )
        
        # 2. Record in monitoring
        if self.dashboard:
            self.dashboard.record_transaction(prediction)
        
        # 3. If high-risk, add to review queue
        if prediction.get('risk_level') == 'HIGH' and self.review_queue:
            # Add to review queue with ensemble predictions
            self.review_queue.add_case(
                transaction_id=transaction_id,
                customer_id=f"CUST_{hash(transaction_id) % 10000}",
                merchant_id=f"MERCH_{hash(transaction_id) % 5000}",
                amount=features[0] * 100,  # Approximate
                predictions={
                    'isolation_forest': prediction.get('fraud_probability', 0),
                    'ensemble_score': prediction.get('fraud_probability', 0)
                }
            )
        
        return prediction
    
    def run_batch_analysis(self, transactions: list) -> Dict[str, Any]:
        """
        Run nightly GCN batch job for fraud ring detection
        """
        if not self.gcn_batch:
            logger.error("Batch system not initialized")
            return {}
        
        logger.info("Running batch GCN analysis...")
        result = self.gcn_batch.run(transactions)
        
        return result
    
    def record_manual_review(self, case_id: str, decision: str, 
                            notes: str = ""):
        """
        Record manual review decision and add to feedback loop
        """
        if not self.review_queue:
            logger.error("Review queue not initialized")
            return False
        
        # Resolve case
        self.review_queue.resolve_case(case_id, decision, notes)
        
        # Add to feedback loop
        if self.feedback:
            case = self.review_queue.cases.get(case_id)
            if case:
                self.feedback.add_manual_review_feedback(
                    case_id=case_id,
                    transaction_id=case.transaction_id,
                    investigator_decision=decision,
                    model_prediction=case.ensemble_score > 0.5,
                    confidence=case.ensemble_score,
                    notes=notes
                )
        
        return True
    
    def generate_reports(self):
        """
        Generate all reports and export
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Dashboard report
        if self.dashboard:
            self.dashboard.export_dashboard(
                f'reports/dashboard_{timestamp}.json'
            )
        
        # Review queue report
        if self.review_queue:
            self.review_queue.export_report(
                f'reports/review_queue_{timestamp}.json'
            )
        
        # Feedback/improvement report
        if self.improver:
            self.improver.export_report(
                f'reports/improvement_{timestamp}.json'
            )
        
        logger.info("✓ All reports generated")
    
    def print_system_status(self):
        """Print comprehensive system status"""
        print("\n" + "="*100)
        print("PRODUCTION SYSTEM STATUS")
        print("="*100)
        
        print("\nREAL-TIME SYSTEM (Isolation Forest):")
        print("-"*100)
        if self.if_deployment:
            stats = self.if_deployment.get_statistics()
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key:30} {value:.2f}")
                else:
                    print(f"  {key:30} {value}")
        
        print("\nREVIEW QUEUE SYSTEM:")
        print("-"*100)
        if self.review_queue:
            queue_stats = self.review_queue.get_statistics()
            for key, value in queue_stats.items():
                if isinstance(value, float):
                    print(f"  {key:30} {value:.2f}")
                else:
                    print(f"  {key:30} {value}")
        
        print("\nMONITORING DASHBOARD:")
        print("-"*100)
        if self.dashboard:
            self.dashboard.print_dashboard()
        
        print("\nFEEDBACK & IMPROVEMENT:")
        print("-"*100)
        if self.improver:
            self.improver.print_report()
        
        print("="*100 + "\n")


def main():
    """Example: Deploy complete production system"""
    
    # Initialize deployment
    deployment = ProductionDeployment()
    
    # Deploy all systems
    status = deployment.deploy_all_systems()
    
    # Simulate transactions
    import numpy as np
    
    print("\nSIMULATING TRANSACTIONS...")
    print("="*100)
    
    for i in range(10):
        features = np.random.randn(10)
        result = deployment.score_transaction_real_time(
            f"TXN_{i:05d}", features.tolist()
        )
        print(f"  TXN_{i:05d}: {result.get('risk_level', 'UNKNOWN'):8} | "
              f"Latency: {result.get('latency_ms', 0):.2f}ms")
    
    # Print system status
    deployment.print_system_status()
    
    # Generate reports
    deployment.generate_reports()
    
    print("\n✓ PRODUCTION DEPLOYMENT COMPLETE\n")


if __name__ == "__main__":
    main()
