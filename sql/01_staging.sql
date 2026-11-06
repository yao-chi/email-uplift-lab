-- 01_staging.sql：原始 CSV -> 带类型、带分群标签的分析宽表
-- 口径说明：
--   * arm 保持原始三分组；is_treated 只用于"任意邮件 vs 不打扰"的合并口径
--   * recency = 距上次购买月数（1-12），history = 过去 12 个月消费额
--   * 所有派生列只用实验前信息，不触碰 visit/conversion/spend
CREATE OR REPLACE TABLE staging AS
SELECT
    CAST(recency AS INTEGER)                                  AS recency,
    history_segment,
    CAST(history AS DOUBLE)                                   AS history,
    CAST(mens AS INTEGER)                                     AS mens,
    CAST(womens AS INTEGER)                                   AS womens,
    zip_code,
    CAST(newbie AS INTEGER)                                   AS newbie,
    channel,
    segment                                                   AS arm,
    CAST(visit AS INTEGER)                                    AS visit,
    CAST(conversion AS INTEGER)                               AS conversion,
    CAST(spend AS DOUBLE)                                     AS spend,
    CASE WHEN segment = 'No E-Mail' THEN 0 ELSE 1 END         AS is_treated,
    CASE WHEN segment = 'Mens E-Mail' THEN 1 ELSE 0 END       AS is_mens,
    CASE WHEN segment = 'Womens E-Mail' THEN 1 ELSE 0 END     AS is_womens,
    CASE WHEN recency BETWEEN 1 AND 3  THEN 'R1_1_3'
         WHEN recency BETWEEN 4 AND 6  THEN 'R2_4_6'
         WHEN recency BETWEEN 7 AND 9  THEN 'R3_7_9'
         ELSE 'R4_10_12' END                                  AS recency_band,
    CASE WHEN history < 100  THEN 'H1_<100'
         WHEN history < 200  THEN 'H2_100_200'
         WHEN history < 350  THEN 'H3_200_350'
         WHEN history < 500  THEN 'H4_350_500'
         WHEN history < 750  THEN 'H5_500_750'
         WHEN history < 1000 THEN 'H6_750_1000'
         ELSE 'H7_1000+' END                                  AS history_band,
    CASE WHEN mens + womens > 1 THEN 1 ELSE 0 END             AS multi_category,
    -- 买过哪一类（既买男装又买女装 = 品类广度最高的客群）
    CASE WHEN mens > 0 AND womens > 0 THEN 'both'
         WHEN mens > 0                  THEN 'mens_only'
         WHEN womens > 0                THEN 'womens_only'
         ELSE 'none' END                                      AS buyer_type,
    -- 用户在实验窗口内的完整漏斗：访问 -> 成交 -> 金额
    CASE WHEN visit = 1 AND conversion = 1 THEN 1 ELSE 0 END   AS funnel_ok
FROM read_csv_auto('__CSV__', header = true);