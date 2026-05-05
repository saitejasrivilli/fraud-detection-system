import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import json


class DataScaler:
    """Manage feature scaling consistently across train/test"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit scaler on training data"""
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform back to original scale"""
        return self.scaler.inverse_transform(X)


def create_sequences(data: np.ndarray, seq_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM from transaction data
    
    Args:
        data: shape (n_transactions, n_features)
        seq_length: window size
    
    Returns:
        X_seq: shape (n_sequences, seq_length, n_features)
        indices: which transactions are end of sequence
    """
    X_seq = []
    indices = []
    
    for i in range(len(data) - seq_length + 1):
        X_seq.append(data[i:i + seq_length])
        indices.append(i + seq_length - 1)
    
    return np.array(X_seq), np.array(indices)


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, 
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
    """
    Time-aware train/test split (important for fraud detection)
    """
    np.random.seed(random_state)
    n = len(X)
    split_idx = int(n * (1 - test_size))
    
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def get_feature_names(n_features: int = 10) -> list:
    """Standard feature names for fraud dataset"""
    return [
        'Amount',
        'Time',
        'Customer_Tx_Count',
        'Customer_Avg_Amount',
        'Customer_Std_Amount',
        'Merchant_Fraud_Rate',
        'Customer_Recency',
        'Amount_Zscore',
        'Hour_of_Day',
        'Day_of_Week'
    ][:n_features]


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics to JSON"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def threshold_optimization(y_true: np.ndarray, y_pred: np.ndarray, 
                           thresholds: np.ndarray = None) -> Dict[str, Any]:
    """
    Find optimal threshold based on Youden's index (sensitivity + specificity - 1)
    """
    from sklearn.metrics import precision_recall_curve, roc_curve
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    youdens = tpr - fpr
    optimal_idx = np.argmax(youdens)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'optimal_threshold': float(optimal_threshold),
        'tpr_at_optimal': float(tpr[optimal_idx]),
        'fpr_at_optimal': float(fpr[optimal_idx])
    }


class PerformanceTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def add(self, epoch: int, train_loss: float, val_loss: float = None):
        """Add epoch metrics"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame(self.history)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")
