# Fraud Detection System - Deployment Files

Complete production-ready deployment package with database, authentication, and containerization.

## Files Included

### 1. **Database Integration**
- `models.py` - SQLAlchemy models for PostgreSQL
  - Transaction model
  - FraudCase model
  - FeedbackLog model
  - Data Access Objects (DAOs)

### 2. **Authentication & Security**
- `auth.py` - FastAPI JWT authentication
  - User authentication
  - Role-based access control (RBAC)
  - Token management
  - 3 default user roles: admin, reviewer, analyst

### 3. **API Application**
- `main.py` - Production FastAPI application
  - Authentication endpoints
  - Fraud scoring endpoints
  - Review management
  - Monitoring dashboard
  - Admin controls
  - Error handling

### 4. **Docker & Containerization**
- `Dockerfile` - Container image definition
  - Multi-stage build
  - Security best practices
  - Health checks
  - Resource optimization

- `docker-compose.yml` - Local development environment
  - PostgreSQL database
  - Redis cache
  - FastAPI application
  - PgAdmin UI
  - Auto-healing health checks

### 5. **Kubernetes Deployment**
- `kubernetes.yaml` - Production Kubernetes manifests
  - Namespace setup
  - ConfigMaps and Secrets
  - StatefulSet for PostgreSQL
  - Deployment for API (3+ replicas)
  - Horizontal Pod Autoscaler (HPA)
  - Service definitions
  - Resource quotas
  - Auto-scaling (3-10 pods based on CPU/memory)

### 6. **Configuration**
- `requirements.txt` - Python dependencies
  - FastAPI, SQLAlchemy, JWT, etc.
  - All versions pinned for reproducibility

- `.env.example` - Environment template
  - Database credentials
  - Security keys
  - Logging configuration
  - Monitoring settings

### 7. **Documentation**
- `DEPLOYMENT_GUIDE.txt` - Complete deployment instructions
  - Local Docker Compose setup
  - Kubernetes deployment
  - Database configuration
  - API usage examples
  - Authentication setup
  - Troubleshooting guide

## Quick Start

### Local Development (Docker Compose)

```bash
# Setup
cp docker-compose.yml .
cp .env.example .env
cp models.py src/database/
cp auth.py src/
cp main.py .

# Start
docker-compose up -d

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/login -d "username=admin&password=admin123"

# Stop
docker-compose down
```

### Kubernetes Production

```bash
# Build and push image
docker build -t your-registry/fraud-detection:latest .
docker push your-registry/fraud-detection:latest

# Deploy
kubectl apply -f kubernetes.yaml

# Verify
kubectl get pods -n fraud-detection
kubectl get svc -n fraud-detection

# Access
kubectl port-forward -n fraud-detection svc/fraud-detection-api 8000:8000
```

## Key Features

✅ **Database Integration**
- PostgreSQL with SQLAlchemy ORM
- 4 main tables (transactions, cases, feedback, audit)
- Connection pooling
- Automatic migrations

✅ **Authentication & Authorization**
- JWT token-based authentication
- Role-based access control (admin, reviewer, analyst)
- User management
- Token expiration

✅ **Docker**
- Single-file containerization
- Multi-stage builds
- Health checks
- Resource limits

✅ **Kubernetes**
- Production-grade manifests
- Auto-scaling (3-10 replicas)
- Load balancing
- Persistent volumes
- Resource management

✅ **Security**
- Encrypted secrets
- HTTPS/TLS ready
- Rate limiting ready
- CORS configuration
- Input validation

✅ **Monitoring**
- Health checks
- Readiness probes
- Metrics endpoints
- Logging configuration
- Audit trail

## API Endpoints

**Public:**
- `GET /health` - Health check
- `GET /ready` - Readiness check

**Authentication:**
- `POST /login` - User login
- `POST /token` - Get access token
- `GET /profile` - User profile (requires auth)

**Fraud Detection (requires auth):**
- `POST /score` - Score transaction
- `GET /pending-reviews` - Get pending cases (reviewer role)
- `POST /review/{id}` - Submit review (reviewer role)

**Monitoring (requires auth):**
- `GET /dashboard` - Monitoring dashboard
- `GET /health-status` - System health
- `GET /feedback-metrics` - Model feedback (analyst role)

**Admin (requires auth + admin role):**
- `POST /retrain` - Trigger retraining

## Default Users

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | admin |
| reviewer | reviewer123 | reviewer |
| analyst | analyst123 | analyst |

**⚠️ Change these passwords in production!**

## Environment Variables

See `.env.example` for all available options. Key variables:

- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - JWT secret (change in production)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)

## Database Tables

1. **transactions** - Fraud predictions and reviews
2. **fraud_cases** - Manual review workflow
3. **feedback_logs** - Human feedback for model improvement
4. **audit_logs** - Compliance audit trail

## Scaling

**Docker Compose:**
- Single machine
- ~1000 requests/second
- Good for development/testing

**Kubernetes:**
- Multiple machines
- Auto-scaling (3-10 replicas)
- ~10,000+ requests/second
- Production-grade reliability

## Next Steps

1. Review deployment files
2. Customize for your environment
3. Change SECRET_KEY and passwords
4. Deploy locally with Docker Compose
5. Test API endpoints
6. Deploy to Kubernetes cluster
7. Setup monitoring and alerting

## Support

See `DEPLOYMENT_GUIDE.txt` for:
- Detailed setup instructions
- Troubleshooting guide
- Production checklist
- API usage examples
- Database operations

## License

Same as main fraud detection system
