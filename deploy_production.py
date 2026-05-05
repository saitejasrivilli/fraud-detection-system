"""Production Deployment Example"""

import numpy as np
from src.production import get_orchestrator
from src.data_prep import create_sample_dataset
from src.utils import split_train_test, DataScaler


def main():
    print("=" * 80)
    print("  PRODUCTION DEPLOYMENT EXAMPLE")
    print("=" * 80)
    
    orchestrator = get_orchestrator()
    
    print("\n1. PREPARING DATA...")
    X, y = create_sample_dataset(n_samples=10000, fraud_rate=0.001)
    scaler = DataScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = split_train_test(X_scaled, y, test_size=0.3)
    X_normal = X_train[y_train == 0]
    print(f"✓ Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test")
    
    print("\n2. DEPLOYING ISOLATION FOREST...")
    orchestrator.if_service.train_and_save(X_normal, "models/isolation_forest.pkl")
    
    print("\n3. REAL-TIME SCORING...")
    for i in range(3):
        features = X_test[i]
        result = orchestrator.score_transaction(features, transaction_id=f"TXN_{i}")
        print(f"\n  Transaction {i}:")
        print(f"    Fraud: {result['fraud_prediction']}")
        print(f"    Score: {result['fraud_probability']:.3f}")
        print(f"    Risk: {result['risk_level']}")
    
    print("\n4. BATCH SCORING...")
    batch_result = orchestrator.score_batch(X_test[:100], batch_id="BATCH_001")
    print(f"  Scored: {batch_result['n_transactions']}")
    print(f"  Fraud: {batch_result['n_fraud_detected']}")
    print(f"  Latency: {batch_result['latency_ms']:.2f}ms")
    
    print("\n5. SCHEDULING OVERNIGHT ANALYSIS...")
    batch_job = orchestrator.schedule_overnight_analysis(X_train, y_train)
    print(f"✓ Job: {batch_job}")
    
    print("\n6. MONITORING DASHBOARD...")
    dashboard = orchestrator.get_dashboard()
    print(f"  Status: {dashboard['system_status']}")
    print(f"  Predictions: {dashboard['metrics']['n_predictions']}")
    
    print("\n7. SYSTEM HEALTH...")
    health = orchestrator.get_system_health()
    print(f"  Health: {health['status']}")
    
    print("\n8. FEEDBACK METRICS...")
    feedback = orchestrator.get_feedback_metrics()
    print(f"  Accuracy: {feedback['retraining_status']['current_accuracy']:.2%}")
    
    print("\n9. DEPLOYMENT STATUS...")
    status = orchestrator.get_deployment_status()
    print(f"  Uptime: {status['uptime_seconds']:.0f}s")
    
    print("\n" + "=" * 80)
    print("✅ PRODUCTION DEPLOYMENT COMPLETE")
    print("=" * 80)
    print(orchestrator.generate_report())


if __name__ == "__main__":
    main()
