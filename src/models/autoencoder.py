import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import time
from typing import Dict, Any, Tuple


class FraudAutoencoder:
    """
    Unsupervised anomaly detection using autoencoder
    Normal transactions learn to reconstruct, fraud has high reconstruction error
    """
    
    def __init__(self, input_dim: int = 10, encoding_dim: int = 16, random_state: int = 42):
        """
        Args:
            input_dim: number of input features
            encoding_dim: bottleneck layer size
            random_state: for reproducibility
        """
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.encoder = None
        self.threshold = None
    
    def build_model(self):
        """Build autoencoder architecture"""
        print("\nBuilding autoencoder architecture...")
        print(f"  Input: {self.input_dim} features")
        print(f"  Bottleneck: {self.encoding_dim} dimensions")
        
        # Encoder
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        
        # Full autoencoder
        self.model = Model(inputs, decoded, name='autoencoder')
        
        # Encoder (for embeddings)
        self.encoder = Model(inputs, encoded, name='encoder')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        print(f"✓ Model built")
        print(f"\nModel Summary:")
        self.model.summary()
    
    def train(self, X_normal: np.ndarray, X_test: np.ndarray = None, 
              y_test: np.ndarray = None, epochs: int = 50, 
              batch_size: int = 32, validation_split: float = 0.2):
        """
        Train autoencoder on normal transactions only
        
        Args:
            X_normal: training data (normal transactions only)
            X_test: test data (for validation with fraud)
            y_test: true labels for test data
            epochs: number of training epochs
            batch_size: batch size
            validation_split: percentage for validation
        """
        if self.model is None:
            self.build_model()
        
        print(f"\nTraining autoencoder...")
        print(f"  Training samples: {len(X_normal)}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        
        # Prepare validation data if provided
        validation_data = None
        if X_test is not None and y_test is not None:
            validation_data = (X_test, X_test)
        
        start = time.time()
        
        history = self.model.fit(
            X_normal, X_normal,
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
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each sample
        Higher error = more likely to be anomalous
        """
        X_reconstructed = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_reconstructed), axis=1)
        return mse
    
    def set_threshold(self, X_val: np.ndarray, percentile: int = 90):
        """
        Set anomaly threshold based on reconstruction error percentile
        on validation data
        
        Args:
            X_val: validation data
            percentile: percentile threshold (default 90th = top 10% anomalous)
        """
        errors = self.get_reconstruction_error(X_val)
        self.threshold = np.percentile(errors, percentile)
        print(f"\nThreshold set to {self.threshold:.4f} (90th percentile)")
        print(f"  This flags top {100-percentile}% as anomalies")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud (1) or normal (0) based on reconstruction error
        
        Args:
            X: feature matrix
        
        Returns:
            predictions: 1 = fraud, 0 = normal
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        errors = self.get_reconstruction_error(X)
        predictions = (errors > self.threshold).astype(int)
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get continuous anomaly scores (reconstruction error)"""
        errors = self.get_reconstruction_error(X)
        # Normalize to 0-1
        if self.threshold is not None:
            scores = np.clip(errors / (self.threshold * 2), 0, 1)
        else:
            scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)
        return scores
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate autoencoder on labeled test data
        """
        print("\nEvaluating autoencoder...")
        
        predictions = self.predict(X)
        scores = self.get_anomaly_scores(X)
        
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
        """Save model to disk"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get learned embeddings from encoder"""
        if self.encoder is None:
            raise ValueError("Encoder not built.")
        return self.encoder.predict(X, verbose=0)
