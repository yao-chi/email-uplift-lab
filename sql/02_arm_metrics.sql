-- 02_arm_metrics.sql：三臂实验的核心指标口径
-- 这张表是后面所有 ATE / 决策计算的唯一数字来源
CREATE OR REPLACE TABLE arm_metrics AS
SELECT
    arm,
    COUNT(*)                                                   AS users,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4)            AS user_share,
    ROUND(AVG(visit), 4)                                       AS visit_rate,
    ROUND(AVG(conversion), 4)                                  AS conv_rate,
    ROUND(SUM(spend) / COUNT(*), 4)                            AS arpu,
    ROUND(SUM(spend) / NULLIF(SUM(conversion), 0), 2)          AS aov_per_converter,
    -- 条件转化率：访问的人里有多少成交（衡量"流量质量"而非"流量多少"）
    ROUND(SUM(conversion) * 1.0 / NULLIF(SUM(visit), 0), 4)    AS conv_given_visit,
    ROUND(AVG(history), 2)                                     AS avg_history,
    ROUND(AVG(recency), 2)                                     AS avg_recency,
    ROUND(AVG(newbie), 4)                                      AS newbie_share,
    SUM(spend)                                                 AS total_spend,
    SUM(conversion)                                            AS total_conversions,
    SUM(visit)                                                 AS total_visits
FROM staging
GROUP BY arm
ORDER BY arm;