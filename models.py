"""
Database Models - PostgreSQL Integration
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/fraud_detection")

engine = create_engine(DATABASE_URL, echo=False, pool_size=20, max_overflow=40)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Transaction(Base):
    """Transaction model"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    customer_id = Column(String, index=True)
    merchant_id = Column(String, index=True)
    amount = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    features = Column(JSON)
    
    isolation_forest_score = Column(Float, nullable=True)
    autoencoder_score = Column(Float, nullable=True)
    lstm_score = Column(Float, nullable=True)
    gcn_score = Column(Float, nullable=True)
    ensemble_score = Column(Float, nullable=True)
    
    fraud_prediction = Column(Boolean, default=False, index=True)
    risk_level = Column(String)
    
    reviewed = Column(Boolean, default=False, index=True)
    review_decision = Column(String, nullable=True)
    reviewer_id = Column(String, nullable=True)
    review_notes = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FraudCase(Base):
    """Fraud case model"""
    __tablename__ = "fraud_cases"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, unique=True, index=True)
    transaction_id = Column(String, index=True)
    priority = Column(String)
    status = Column(String, index=True)
    assigned_to = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime, nullable=True)
    review_decision = Column(String, nullable=True)
    review_notes = Column(String, nullable=True)


class FeedbackLog(Base):
    """Feedback model"""
    __tablename__ = "feedback_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    transaction_id = Column(String, index=True)
    model_prediction = Column(Boolean)
    human_decision = Column(Boolean)
    fraud_score = Column(Float)
    risk_level = Column(String)
    reviewer_id = Column(String)
    notes = Column(String, nullable=True)
    is_correct = Column(Boolean, index=True)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


class TransactionDAO:
    """Transaction data access"""
    
    @staticmethod
    def create(db, data: dict):
        t = Transaction(**data)
        db.add(t)
        db.commit()
        db.refresh(t)
        return t
    
    @staticmethod
    def get_by_id(db, transaction_id: str):
        return db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    
    @staticmethod
    def get_pending_review(db, limit: int = 10):
        return db.query(Transaction).filter(
            Transaction.reviewed == False,
            Transaction.fraud_prediction == True
        ).order_by(Transaction.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def update_scores(db, transaction_id: str, scores: dict):
        db.query(Transaction).filter(Transaction.transaction_id == transaction_id).update(scores)
        db.commit()
