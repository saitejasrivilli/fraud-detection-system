# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Clone repo
git clone <your-repo-url>
cd fraud-detection-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Full Pipeline (10-15 minutes)

```bash
# Run complete training and evaluation
python main.py
```

**What this does**:
- Generates synthetic fraud dataset (10,000 transactions)
- Trains 6 models (Isolation Forest, Autoencoder, LOF, LSTM, GCN, Ensemble)
- Evaluates all models
- Generates visualizations (ROC curves, PR curves, confusion matrices)
- Prints comparison table and recommendations
- Saves results to `results/` folder

**Output**:
```
================================================================================
  FRAUD DETECTION SYSTEM - MAIN PIPELINE
================================================================================

Dataset Summary:
  Total transactions: 10000
  Features: 10
  Fraud samples: 9 (0.09%)
  Normal samples: 9991 (99.91%)

Train/Test Split:
  Train: 7000 samples (0.09% fraud)
  Test:  3000 samples (0.09% fraud)

================================================================================
  COMPONENT 2: TRADITIONAL ANOMALY DETECTION
================================================================================

Training anomaly detection models...
...
Ensemble Evaluation:
  Precision: 0.930
  Recall: 0.755
  F1: 0.835
  AUC-ROC: 0.868
```

## Running Individual Models

### 1. Isolation Forest Only
```python
from src.models.isolation_forest import AnomalyDetectionEnsemble
from src.data_prep import create_sample_dataset
from src.utils import split_train_test, DataScaler
import numpy as np

# Load data
X, y = create_sample_dataset(n_samples=10000)
scaler = DataScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)

# Train
model = AnomalyDetectionEnsemble()
model.train(X_train[y_train == 0])

# Evaluate
results = model.evaluate(X_test, y_test)
```

### 2. Autoencoder
```python
from src.models.autoencoder import FraudAutoencoder

# Build and train
autoencoder = FraudAutoencoder(input_dim=10)
autoencoder.build_model()
autoencoder.train(X_train[y_train == 0], epochs=30)
autoencoder.set_threshold(X_test, percentile=90)

# Evaluate
results = autoencoder.evaluate(X_test, y_test)
```

### 3. LSTM
```python
from src.models.lstm import FraudLSTM
from src.utils import create_sequences

# Create sequences
lstm = FraudLSTM(seq_length=30, n_features=10)
X_train_seq, _ = create_sequences(X_train[y_train == 0])
X_test_seq, _ = create_sequences(X_test)

# Train
lstm.build_model()
lstm.train(X_train_seq, epochs=20)
lstm.set_threshold(X_test_seq)

# Evaluate
results = lstm.evaluate(X_test_seq, y_test[29:])
```

### 4. GCN
```python
from src.models.gcn import FraudGCN

# Build network
gcn = FraudGCN(n_node_features=5)
gcn.build_graph(X_train, y_train)

# Train
gcn.build_model()
gcn.train(epochs=20)

# Evaluate
results = gcn.evaluate(X_test[:100], y_test[:100])
```

## Starting the Real-Time API

```bash
# Terminal 1: Start API server
python -m uvicorn src.streaming:app --reload

# Terminal 2: Test API
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST123",
    "merchant_id": "MERCH456",
    "amount": 1500.50,
    "timestamp": "2024-01-15T10:30:00"
  }'
```

**Expected response**:
```json
{
  "customer_id": "CUST123",
  "fraud_score": 0.87,
  "fraud_probability": 0.87,
  "is_fraud": true,
  "latency_ms": 12.5,
  "models_agree": 4,
  "models_total": 5,
  "risk_level": "HIGH",
  "reason": "Unusually high amount | Multiple models flagged"
}
```

## Batch Scoring

```bash
curl -X POST "http://localhost:8000/score_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "customer_id": "CUST123",
        "merchant_id": "MERCH456",
        "amount": 100,
        "timestamp": "2024-01-15T10:30:00"
      },
      {
        "customer_id": "CUST124",
        "merchant_id": "MERCH457",
        "amount": 5000,
        "timestamp": "2024-01-15T10:31:00"
      }
    ]
  }'
```

## Monitoring API

```bash
# Get pipeline statistics
curl http://localhost:8000/statistics

# Get recent fraud alerts
curl http://localhost:8000/alerts?limit=10
```

## Using with Real Data (Kaggle)

```python
from src.data_prep import FraudDataPrep

# Download creditcard.csv from Kaggle
# Place in data/creditcard.csv

# Load real data
data_prep = FraudDataPrep()
df = data_prep.load_csv('data/creditcard.csv')

# Create database
data_prep.create_sqlite_db()

# Run feature engineering SQL
X, y, metadata = data_prep.prepare_features()

# Rest of pipeline same as above
```

## Project Structure

```
fraud-detection-system/
├── main.py                 ← Run this for complete pipeline
├── requirements.txt        ← pip install -r requirements.txt
├── README.md              ← Full documentation
├── PROJECT_ANALYSIS.md    ← Deep dive into code
├── QUICKSTART.md          ← This file
├── src/
│   ├── data_prep.py       ← Data loading & SQL
│   ├── evaluation.py      ← Metrics & visualization
│   ├── streaming.py       ← FastAPI real-time API
│   ├── utils.py           ← Helper functions
│   └── models/
│       ├── isolation_forest.py
│       ├── autoencoder.py
│       ├── lstm.py
│       └── gcn.py
├── sql/
│   └── feature_engineering.sql
├── results/               ← Generated outputs
│   ├── model_comparison.csv
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── confusion_matrices.png
└── notebooks/             ← Jupyter notebooks (optional)
    ├── 01_eda.ipynb
    ├── 02_isolation_forest.ipynb
    └── ...
```

## Troubleshooting

### ImportError: No module named 'tensorflow'
```bash
pip install tensorflow keras
```

### CUDA/GPU issues (if you want GPU acceleration)
```bash
# CPU only (default, works fine)
pip install tensorflow-cpu

# Or with GPU
pip install tensorflow[and-cuda]
```

### Out of memory
```python
# In main.py, reduce dataset size
X, y = create_sample_dataset(n_samples=5000)  # Instead of 10000
```

### Plots not showing (in script)
- Plots are automatically saved to `results/` folder
- View them with an image viewer
- Or run code in Jupyter notebook for interactive plots

### Port 8000 already in use
```bash
# Use different port
python -m uvicorn src.streaming:app --reload --port 8001
```

## Key Files to Understand

1. **Start here**: `README.md` - Overview and architecture
2. **Then read**: `PROJECT_ANALYSIS.md` - Deep dive on each component
3. **Code walkthrough**:
   - `src/data_prep.py` - Understand data flow
   - `src/models/isolation_forest.py` - Simplest model
   - `src/models/autoencoder.py` - Deep learning
   - `src/models/gcn.py` - Most advanced
4. **Integration**: `main.py` - How everything fits together
5. **API**: `src/streaming.py` - Production serving

## Interview Prep Checklist

- [ ] Run `main.py` successfully
- [ ] Understand each model (what it does, why it works)
- [ ] Review the talking points in `PROJECT_ANALYSIS.md`
- [ ] Practice explaining architecture in 2 minutes
- [ ] Be ready to discuss trade-offs (precision vs recall, latency vs accuracy)
- [ ] Have repo link ready
- [ ] Prepare "walk me through your code" answers
- [ ] Study SQL queries in `sql/feature_engineering.sql`
- [ ] Test the API manually (curl or Postman)
- [ ] Review results (ROC curves, confusion matrices)

## Next Steps

1. ✅ Get code running locally
2. ✅ Understand each component
3. ✅ Practice explaining it
4. ✅ Push to GitHub
5. ✅ Link in job applications
6. ✅ Practice interview questions
7. ✅ Be ready to discuss extensions/improvements

## Common Interview Questions

**Q: Why multiple models instead of one?**
A: Different fraud signals. IF catches anomalies, Autoencoder catches patterns, GCN catches rings, LSTM catches sequences. Ensemble combines them.

**Q: How would you scale this to Uber's size?**
A: Real-time path (IF < 1ms), batch path (GCN overnight), ensemble for reviews. Add: caching, distributed processing, A/B testing.

**Q: How do you handle class imbalance?**
A: Train unsupervised on normal only. Use ensemble voting. Threshold optimization. Time-series split (no shuffle).

**Q: What if fraud patterns change?**
A: Monitor model performance. Retrain weekly. Implement feedback loop. Alert on disagreement between models.

**Q: How would you explain a fraud decision to customer?**
A: Use feature importance. "Amount unusual for you" vs "Same merchant as known fraudster" vs "Rapid transactions".

---

**You've got this. Good luck!** 🚀
