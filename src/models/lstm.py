import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import time
from typing import Dict, Any, Tuple


class FraudLSTM:
    """
    Time-series LSTM for detecting unusual transaction sequences
    Learns normal transaction patterns, flags unusual sequences
    """
    
    def __init__(self, seq_length: int = 30, n_features: int = 3, random_state: int = 42):
        """
        Args:
            seq_length: number of timesteps (transactions) in sequence
            n_features: features per transaction
            random_state: reproducibility
        """
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.threshold = None
    
    def build_model(self):
        """Build LSTM architecture for sequence classification"""
        print("\nBuilding LSTM architecture...")
        print(f"  Sequence length: {self.seq_length} transactions")
        print(f"  Features per transaction: {self.n_features}")
        
        model = keras.Sequential([
            layers.Input(shape=(self.seq_length, self.n_features)),
            
            # LSTM layers
            layers.LSTM(64, return_sequences=True, activation='relu',
                       dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(32, return_sequences=False, activation='relu',
                       dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers for classification
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"✓ Model built")
        print(f"\nModel Summary:")
        self.model.summary()
    
    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create sequences from flat transaction data
        
        Args:
            X: shape (n_transactions, n_features)
        
        Returns:
            sequences: shape (n_sequences, seq_length, n_features)
        """
        sequences = []
        
        for i in range(len(X) - self.seq_length + 1):
            seq = X[i:i + self.seq_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def train(self, X_normal: np.ndarray, X_test: np.ndarray = None,
              y_test: np.ndarray = None, epochs: int = 50,
              batch_size: int = 32, validation_split: float = 0.2):
        """
        Train LSTM on normal transaction sequences
        
        Args:
            X_normal: training sequences (normal only)
            X_test: test sequences
            y_test: test labels
            epochs: training epochs
            batch_size: batch size
            validation_split: validation split
        """
        if self.model is None:
            self.build_model()
        
        print(f"\nTraining LSTM...")
        print(f"  Training sequences: {len(X_normal)}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        
        validation_data = None
        if X_test is not None and y_test is not None:
            validation_data = (X_test, y_test)
        
        start = time.time()
        
        history = self.model.fit(
            X_normal, np.zeros(len(X_normal)),  # Train on normal = 0 (not fraud)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            verbose=1,
            shuffle=True
        )
        
        elapsed = time.time() - start
        print(f"✓ Training complete in {elapsed:.2f}s")
        
        return history
    
    def get_fraud_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get fraud probability for each sequence
        
        Args:
            X: sequences of shape (n_sequences, seq_length, n_features)
        
        Returns:
            probabilities: shape (n_sequences,) with values in [0, 1]
        """
        probs = self.model.predict(X, verbose=0)
        return probs.flatten()
    
    def set_threshold(self, X_val: np.ndarray, percentile: int = 90):
        """
        Set fraud threshold based on fraud probability percentile
        
        Args:
            X_val: validation sequences
            percentile: percentile threshold
        """
        probs = self.get_fraud_probabilities(X_val)
        self.threshold = np.percentile(probs, percentile)
        print(f"\nThreshold set to {self.threshold:.4f} (90th percentile)")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud (1) or normal (0)
        
        Args:
            X: sequences
        
        Returns:
            predictions: 1 = fraud, 0 = normal
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        probs = self.get_fraud_probabilities(X)
        predictions = (probs > self.threshold).astype(int)
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate LSTM on test sequences with labels
        """
        print("\nEvaluating LSTM...")
        
        predictions = self.predict(X)
        scores = self.get_fraud_probabilities(X)
        
        results = {
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'auc_roc': roc_auc_score(y, scores),
        }
        
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1:        {results['f1']:.3f}")
        print(f"  AUC-ROC:   {results['auc_roc']:.3f}")
        
        return results
    
    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
