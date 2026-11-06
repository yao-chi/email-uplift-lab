-- 03_segment_metrics.sql：把宽表 unpivot 成"维度-取值-分组"长表
-- 目的：一次 SQL 产出所有分群口径，避免每个维度写一遍 group by
-- 注意：multi_category 是整数，unpivot 前必须显式转成 VARCHAR，否则 DuckDB 报
--       "Cannot unpivot columns of types VARCHAR and INTEGER"
--       UNPIVOT 会自动保留未被 unpivot 的列（arm / visit / conversion / spend），
--       所以这里不需要再 join 回 staging。
CREATE OR REPLACE VIEW seg_base AS
SELECT
    arm,
    visit,
    conversion,
    spend,
    recency_band,
    history_band,
    channel,
    zip_code,
    buyer_type,
    CAST(multi_category AS VARCHAR) AS multi_category
FROM staging;

CREATE OR REPLACE TABLE segment_metrics AS
WITH long AS (
    UNPIVOT seg_base
    ON recency_band, history_band, channel, zip_code, buyer_type, multi_category
    INTO NAME dimension VALUE segment_value
)
SELECT
    dimension,
    CAST(segment_value AS VARCHAR)                             AS segment_value,
    arm,
    COUNT(*)                                                   AS users,
    ROUND(AVG(visit), 4)                                       AS visit_rate,
    ROUND(AVG(conversion), 4)                                  AS conv_rate,
    ROUND(SUM(spend) / COUNT(*), 4)                            AS arpu,
    SUM(visit)                                                 AS visits,
    SUM(conversion)                                            AS conversions
FROM long
GROUP BY dimension, segment_value, arm
ORDER BY dimension, segment_value, arm;