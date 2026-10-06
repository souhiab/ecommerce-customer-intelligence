-- Customer-level analytical features from synthetic transaction tables.
-- SQLite-compatible and intentionally readable for interview discussion.

WITH order_history AS (
    SELECT
        customer_id,
        MIN(order_date) AS first_purchase_date,
        MAX(order_date) AS last_purchase_date,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(total_amount) AS monetary_value,
        AVG(total_amount) AS average_order_value
    FROM orders
    GROUP BY customer_id
),
category_counts AS (
    SELECT
        o.customer_id,
        p.category,
        SUM(oi.quantity) AS units_purchased,
        ROW_NUMBER() OVER (
            PARTITION BY o.customer_id
            ORDER BY SUM(oi.quantity) DESC, p.category
        ) AS category_rank
    FROM orders AS o
    JOIN order_items AS oi ON o.order_id = oi.order_id
    JOIN products AS p ON oi.product_id = p.product_id
    GROUP BY o.customer_id, p.category
),
category_features AS (
    SELECT
        customer_id,
        COUNT(*) AS unique_categories,
        MAX(CASE WHEN category_rank = 1 THEN category END) AS favorite_category
    FROM category_counts
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    CAST(julianday('2025-12-31') - julianday(oh.last_purchase_date) AS INTEGER) AS recency_days,
    oh.frequency,
    ROUND(oh.monetary_value, 2) AS monetary_value,
    ROUND(oh.average_order_value, 2) AS average_order_value,
    CAST(julianday('2025-12-31') - julianday(oh.first_purchase_date) AS INTEGER) AS days_since_first_purchase,
    CASE
        WHEN oh.frequency > 1 THEN ROUND(
            (julianday(oh.last_purchase_date) - julianday(oh.first_purchase_date))
            / (oh.frequency - 1), 1
        )
        ELSE NULL
    END AS average_purchase_interval,
    cf.unique_categories,
    cf.favorite_category,
    oh.last_purchase_date,
    oh.first_purchase_date
FROM customers AS c
JOIN order_history AS oh ON c.customer_id = oh.customer_id
LEFT JOIN category_features AS cf ON c.customer_id = cf.customer_id
ORDER BY c.customer_id;
