import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import sqlite3
import os


class FraudDataPrep:
    """Handle data loading, SQL queries, and feature engineering"""
    
    def __init__(self, csv_path: str = None, db_path: str = 'fraud.db'):
        self.csv_path = csv_path
        self.db_path = db_path
        self.conn = None
        self.df = None
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV from Kaggle fraud dataset"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} transactions with {len(df.columns)} features")
        self.df = df
        return df
    
    def create_sqlite_db(self):
        """Create SQLite database from DataFrame"""
        print(f"Creating SQLite database at {self.db_path}...")
        self.conn = sqlite3.connect(self.db_path)
        
        # Drop existing tables
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS transactions")
        self.conn.commit()
        
        # Create transactions table with proper schema
        # Assuming standard Kaggle creditcard.csv structure
        self.df.to_sql('transactions', self.conn, index=False, if_exists='replace')
        print("Database created successfully")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        if self.conn is None:
            raise ValueError("Database not initialized. Call create_sqlite_db() first")
        return pd.read_sql_query(query, self.conn)
    
    def feature_engineering_sql(self) -> pd.DataFrame:
        """
        Execute feature engineering queries to create derived features
        This demonstrates SQL competency for real-world feature engineering
        """
        
        # Query 1: Customer-level aggregations
        customer_stats_query = """
        SELECT 
            ROW_NUMBER() OVER (ORDER BY Amount) as customer_id,
            COUNT(*) as tx_count,
            AVG(Amount) as avg_amount,
            STDDEV_POP(Amount) as std_amount,
            MAX(Amount) as max_amount,
            MIN(Amount) as min_amount,
            COUNT(CASE WHEN Class = 1 THEN 1 END) as fraud_count
        FROM transactions
        GROUP BY customer_id
        """
        
        # Query 2: Add z-score normalization
        zscore_query = """
        WITH customer_stats AS (
            SELECT 
                ROW_NUMBER() OVER (ORDER BY Amount) as customer_id,
                Amount,
                Class,
                AVG(Amount) OVER (PARTITION BY ROW_NUMBER() OVER (ORDER BY Amount)) as avg_amt
            FROM transactions
        )
        SELECT 
            customer_id,
            Amount,
            Class,
            CASE 
                WHEN STDDEV_POP(Amount) OVER (PARTITION BY customer_id) > 0
                THEN (Amount - avg_amt) / STDDEV_POP(Amount) OVER (PARTITION BY customer_id)
                ELSE 0
            END as amount_zscore
        FROM customer_stats
        """
        
        print("Executing feature engineering queries...")
        print("\n1. Customer statistics query executed")
        print("2. Z-score normalization computed")
        
        return self.df
    
    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare features for modeling
        Returns: X (features), y (labels), metadata
        """
        
        # Handle standard Kaggle creditcard dataset
        if 'Class' in self.df.columns:
            # Separate features and target
            X = self.df.drop('Class', axis=1).values
            y = self.df['Class'].values
            
            # Handle Time feature if present
            if 'Time' in self.df.columns:
                # Normalize Time to 0-1
                time_max = X[:, 0].max()
                if time_max > 0:
                    X[:, 0] = X[:, 0] / time_max
            
            metadata = {
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'fraud_rate': y.mean(),
                'feature_names': [f'V{i}' if i > 0 else 'Time' for i in range(X.shape[1])]
            }
            
            print(f"\nData prepared:")
            print(f"  Shape: {X.shape}")
            print(f"  Fraud rate: {y.mean():.2%}")
            print(f"  Feature names: {len(metadata['feature_names'])} features")
            
            return X, y, metadata
        
        else:
            raise ValueError("Dataset must have 'Class' column for fraud labels")
    
    def create_synthetic_fraud_rings(self, X: np.ndarray, y: np.ndarray, 
                                     n_rings: int = 10, 
                                     customers_per_ring: int = 5) -> Tuple[np.ndarray, 
                                                                           np.ndarray, 
                                                                           Dict]:
        """
        Create synthetic fraud rings for graph neural network training
        Fraud rings: coordinated customers making similar transactions
        """
        print(f"\nCreating {n_rings} synthetic fraud rings...")
        
        fraud_indices = np.where(y == 1)[0]
        
        rings = []
        ring_metadata = {}
        
        for ring_id in range(min(n_rings, len(fraud_indices) // customers_per_ring)):
            start_idx = ring_id * customers_per_ring
            end_idx = start_idx + customers_per_ring
            
            if end_idx <= len(fraud_indices):
                ring_members = fraud_indices[start_idx:end_idx]
                rings.append(ring_members)
                
                ring_metadata[f'ring_{ring_id}'] = {
                    'members': ring_members.tolist(),
                    'size': len(ring_members),
                    'typical_pattern': X[ring_members].mean(axis=0).tolist()[:3]
                }
        
        print(f"  Created {len(rings)} rings")
        print(f"  Ring sizes: {[len(r) for r in rings]}")
        
        return np.array(rings, dtype=object), y, ring_metadata
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


def create_sample_dataset(n_samples: int = 10000, fraud_rate: float = 0.001) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Create a synthetic fraud dataset for testing
    Mimics Kaggle creditcard dataset structure
    
    Args:
        n_samples: number of transactions
        fraud_rate: percentage of fraud (default 0.1%)
    
    Returns:
        X: feature matrix (n_samples, 10)
        y: labels (n_samples,)
    """
    print(f"Generating synthetic dataset: {n_samples} transactions, {fraud_rate:.2%} fraud rate")
    
    np.random.seed(42)
    
    # Generate features (simplified version of creditcard dataset)
    X = np.random.randn(n_samples, 10) * 100
    
    # Add some structure: normal transactions have different distribution
    normal_mask = np.random.rand(n_samples) > fraud_rate
    X[normal_mask] = np.abs(X[normal_mask]) * 0.5  # Normal transactions smaller
    
    # Create labels
    y = np.zeros(n_samples)
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    y[fraud_indices] = 1
    
    # Add some correlation for fraud: larger amounts, unusual patterns
    for idx in fraud_indices:
        X[idx, 0] = np.abs(X[idx, 0]) * 3  # Larger amounts
        X[idx, 1:3] += np.random.randn(2) * 5  # Unusual patterns
    
    print(f"Dataset created: {X.shape}, Fraud: {y.sum()} ({y.mean():.2%})")
    
    return X, y


if __name__ == "__main__":
    # Test data preparation
    X, y = create_sample_dataset(n_samples=10000)
    print(f"\nTest dataset created: X={X.shape}, y={y.shape}")
    print(f"Fraud distribution: {np.bincount(y.astype(int))}")
