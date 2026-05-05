-- SQL Feature Engineering for Fraud Detection
-- These queries demonstrate SQL competency for feature engineering at scale

-- ============================================================================
-- 1. CUSTOMER-LEVEL AGGREGATIONS
-- ============================================================================

-- Basic customer statistics
SELECT 
    customer_id,
    COUNT(*) as tx_count,
    AVG(amount) as avg_amount,
    STDDEV_POP(amount) as std_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount,
    SUM(amount) as total_spent,
    COUNT(DISTINCT merchant_id) as unique_merchants,
    COUNT(CASE WHEN Class = 1 THEN 1 END) as fraud_count,
    COUNT(CASE WHEN Class = 1 THEN 1 END) * 100.0 / COUNT(*) as fraud_rate
FROM transactions
GROUP BY customer_id
ORDER BY fraud_count DESC;


-- ============================================================================
-- 2. TIME-BASED FEATURES
-- ============================================================================

-- Transactions per hour and day patterns
SELECT 
    customer_id,
    EXTRACT(HOUR FROM Time) as hour_of_day,
    EXTRACT(DAY FROM Time) as day_of_month,
    COUNT(*) as tx_count_by_hour,
    AVG(amount) as avg_amount_by_hour,
    COUNT(CASE WHEN Class = 1 THEN 1 END) as fraud_count_by_hour
FROM transactions
GROUP BY customer_id, EXTRACT(HOUR FROM Time), EXTRACT(DAY FROM Time)
ORDER BY customer_id, hour_of_day;


-- ============================================================================
-- 3. MERCHANT-LEVEL FEATURES
-- ============================================================================

-- Merchant risk profile
SELECT 
    merchant_id,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(*) as total_transactions,
    AVG(amount) as avg_tx_amount,
    COUNT(CASE WHEN Class = 1 THEN 1 END) as fraud_count,
    COUNT(CASE WHEN Class = 1 THEN 1 END) * 100.0 / COUNT(*) as merchant_fraud_rate,
    STDDEV_POP(amount) as amount_volatility
FROM transactions
GROUP BY merchant_id
HAVING COUNT(*) > 100  -- Only merchants with significant volume
ORDER BY merchant_fraud_rate DESC;


-- ============================================================================
-- 4. AMOUNT DEVIATION FROM HISTORICAL
-- ============================================================================

-- Z-score of transaction amount vs customer's historical average
WITH customer_stats AS (
    SELECT 
        customer_id,
        AVG(amount) as avg_amt,
        STDDEV_POP(amount) as std_amt
    FROM transactions
    GROUP BY customer_id
)
SELECT 
    t.customer_id,
    t.amount,
    t.Class,
    cs.avg_amt,
    cs.std_amt,
    CASE 
        WHEN cs.std_amt > 0 
        THEN (t.amount - cs.avg_amt) / cs.std_amt
        ELSE 0
    END as amount_zscore,
    CASE 
        WHEN cs.std_amt > 0 AND ABS((t.amount - cs.avg_amt) / cs.std_amt) > 3
        THEN 1 
        ELSE 0
    END as is_outlier_amount
FROM transactions t
JOIN customer_stats cs ON t.customer_id = cs.customer_id
ORDER BY ABS((t.amount - cs.avg_amt) / cs.std_amt) DESC
LIMIT 1000;


-- ============================================================================
-- 5. RECENCY FEATURES (TIME SINCE LAST TRANSACTION)
-- ============================================================================

-- Days since last transaction (recency)
WITH ranked_transactions AS (
    SELECT 
        customer_id,
        Time,
        LAG(Time) OVER (PARTITION BY customer_id ORDER BY Time) as prev_tx_time
    FROM transactions
)
SELECT 
    customer_id,
    Time,
    CAST(Time - prev_tx_time AS FLOAT) as hours_since_last_tx,
    CASE 
        WHEN Time - prev_tx_time < 24 THEN 'Within 24h'
        WHEN Time - prev_tx_time < 168 THEN 'Within 1 week'
        WHEN Time - prev_tx_time < 720 THEN 'Within 1 month'
        ELSE 'Over 1 month'
    END as recency_bucket
FROM ranked_transactions
WHERE prev_tx_time IS NOT NULL
ORDER BY customer_id, Time;


-- ============================================================================
-- 6. CUSTOMER VELOCITY (TRANSACTION FREQUENCY)
-- ============================================================================

-- How many transactions in last 7 days, 30 days, 90 days
SELECT 
    customer_id,
    Time,
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY Time 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_tx_count,
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY Time
        ROWS BETWEEN 168 PRECEDING AND CURRENT ROW  -- 7 days
    ) as tx_count_7d,
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY Time
        ROWS BETWEEN 720 PRECEDING AND CURRENT ROW  -- 30 days
    ) as tx_count_30d,
    COUNT(*) OVER (
        PARTITION BY customer_id 
        ORDER BY Time
        ROWS BETWEEN 2160 PRECEDING AND CURRENT ROW  -- 90 days
    ) as tx_count_90d,
    Class
FROM transactions
ORDER BY customer_id, Time;


-- ============================================================================
-- 7. CUSTOMER LIFETIME VALUE (CLV)
-- ============================================================================

-- Total spend, average spend, and categorization
WITH customer_spend AS (
    SELECT 
        customer_id,
        SUM(amount) as total_spent,
        AVG(amount) as avg_spent,
        COUNT(*) as lifetime_txs,
        MIN(Time) as first_tx_time,
        MAX(Time) as last_tx_time,
        MAX(Time) - MIN(Time) as customer_age_days
    FROM transactions
    GROUP BY customer_id
)
SELECT 
    customer_id,
    total_spent,
    avg_spent,
    lifetime_txs,
    customer_age_days,
    NTILE(4) OVER (ORDER BY total_spent) as spend_quartile,
    CASE 
        WHEN total_spent > PERCENTILE_CONT(0.75) OVER () THEN 'High Value'
        WHEN total_spent > PERCENTILE_CONT(0.50) OVER () THEN 'Medium Value'
        WHEN total_spent > PERCENTILE_CONT(0.25) OVER () THEN 'Low Value'
        ELSE 'Minimal Value'
    END as clv_segment
FROM customer_spend
ORDER BY total_spent DESC;


-- ============================================================================
-- 8. MERCHANT CO-OCCURRENCE (FOR FRAUD RINGS)
-- ============================================================================

-- Customers who use the same merchants (potential fraud rings)
WITH merchant_customers AS (
    SELECT 
        merchant_id,
        customer_id
    FROM transactions
    GROUP BY merchant_id, customer_id
)
SELECT 
    m1.customer_id as customer_1,
    m2.customer_id as customer_2,
    COUNT(DISTINCT m1.merchant_id) as shared_merchants,
    COUNT(DISTINCT m1.merchant_id) * 100.0 / 
        (SELECT COUNT(DISTINCT merchant_id) FROM transactions 
         WHERE customer_id IN (m1.customer_id, m2.customer_id)) as shared_merchant_ratio
FROM merchant_customers m1
JOIN merchant_customers m2 ON m1.merchant_id = m2.merchant_id
    AND m1.customer_id < m2.customer_id
GROUP BY m1.customer_id, m2.customer_id
HAVING COUNT(DISTINCT m1.merchant_id) >= 3  -- At least 3 shared merchants
ORDER BY shared_merchants DESC;


-- ============================================================================
-- 9. TRANSACTION PATTERN CONSISTENCY
-- ============================================================================

-- Variability in transaction amounts and timing (fraud often irregular)
WITH customer_patterns AS (
    SELECT 
        customer_id,
        STDDEV_POP(amount) as amount_std,
        AVG(amount) as amount_mean,
        AVG(amount) / NULLIF(STDDEV_POP(amount), 0) as coefficient_of_variation,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) as q1_amount,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) as q3_amount
    FROM transactions
    GROUP BY customer_id
)
SELECT 
    customer_id,
    amount_mean,
    amount_std,
    coefficient_of_variation,
    q3_amount - q1_amount as iqr_amount,
    CASE 
        WHEN coefficient_of_variation > 1.5 THEN 'High Variability'
        WHEN coefficient_of_variation > 1.0 THEN 'Medium Variability'
        ELSE 'Low Variability'
    END as pattern_consistency
FROM customer_patterns
ORDER BY coefficient_of_variation DESC;


-- ============================================================================
-- 10. FEATURE ENGINEERING PIPELINE (COMPLETE FEATURE TABLE)
-- ============================================================================

-- Create comprehensive feature table for ML model
WITH customer_features AS (
    SELECT 
        customer_id,
        COUNT(*) as tx_count,
        AVG(amount) as avg_amount,
        STDDEV_POP(amount) as std_amount,
        MAX(amount) as max_amount,
        COUNT(DISTINCT merchant_id) as unique_merchants,
        COUNT(CASE WHEN Class = 1 THEN 1 END) as fraud_count,
        COUNT(CASE WHEN Class = 1 THEN 1 END) * 100.0 / COUNT(*) as fraud_rate
    FROM transactions
    GROUP BY customer_id
),
merchant_features AS (
    SELECT 
        merchant_id,
        COUNT(*) as merchant_tx_count,
        AVG(amount) as merchant_avg_amount,
        COUNT(CASE WHEN Class = 1 THEN 1 END) * 100.0 / COUNT(*) as merchant_fraud_rate
    FROM transactions
    GROUP BY merchant_id
)
SELECT 
    t.customer_id,
    t.merchant_id,
    t.amount,
    t.Time,
    t.Class as is_fraud,
    cf.tx_count,
    cf.avg_amount,
    cf.std_amount,
    cf.max_amount,
    cf.unique_merchants,
    cf.fraud_rate as customer_fraud_rate,
    mf.merchant_tx_count,
    mf.merchant_avg_amount,
    mf.merchant_fraud_rate,
    CASE 
        WHEN cf.std_amount > 0
        THEN (t.amount - cf.avg_amount) / cf.std_amount
        ELSE 0
    END as amount_zscore,
    EXTRACT(HOUR FROM t.Time) as hour_of_day,
    EXTRACT(DAY FROM CAST(t.Time as DATE)) as day_of_week
FROM transactions t
LEFT JOIN customer_features cf ON t.customer_id = cf.customer_id
LEFT JOIN merchant_features mf ON t.merchant_id = mf.merchant_id
WHERE t.Class = 1 OR RANDOM() < 0.01  -- Include all fraud + 1% of normal for balance
LIMIT 50000;


-- ============================================================================
-- NOTES FOR PRODUCTION
-- ============================================================================

/*
1. INDEXING: Add indexes on frequently filtered columns
   CREATE INDEX idx_customer_time ON transactions(customer_id, Time);
   CREATE INDEX idx_merchant_class ON transactions(merchant_id, Class);

2. PARTITIONING: For large tables, partition by date
   CREATE TABLE transactions_2024_01 PARTITION OF transactions
   FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

3. MATERIALIZED VIEWS: Pre-compute expensive features
   CREATE MATERIALIZED VIEW customer_stats AS (...)
   REFRESH MATERIALIZED VIEW CONCURRENTLY customer_stats;

4. INCREMENTAL UPDATES: In production, use incremental logic
   - Only process new transactions since last run
   - Update statistics hourly, not daily
   - Cache customer profiles in Redis for sub-second lookups

5. SCALABILITY: For billions of transactions
   - Use distributed SQL (BigQuery, Redshift, Snowflake)
   - Window functions scale better than self-joins
   - Partition early and often
*/
