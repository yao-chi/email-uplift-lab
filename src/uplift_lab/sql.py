"""SQL 口径层（DuckDB）。

为什么要有这一层：指标口径必须写成可复盘的 SQL，而不是散落在 Python 里的
groupby —— 面试官问"你这个留存量怎么算的"，你能直接打开 02_arm_metrics.sql。

DuckDB 的选择理由：单文件、零服务、支持窗口函数/CTE，语法与 MySQL 高度接近。
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import RAW_CSV, SQL_DIR

SCRIPTS = ["01_staging.sql", "02_arm_metrics.sql", "03_segment_metrics.sql"]


def connect(csv_path: Path = RAW_CSV) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET preserve_insertion_order = true")
    # 让 read_csv_auto 用统一路径写法（Windows 反斜杠在 SQL 字符串里要转义）
    con.execute(f"SET VARIABLE csv_path = '{str(csv_path).replace(chr(92), '/')}'")
    return con


def run_script(con: duckdb.DuckDBPyConnection, path: Path, csv_path: Path = RAW_CSV) -> None:
    sql = path.read_text(encoding="utf-8")
    sql = sql.replace("__CSV__", str(csv_path).replace("\\", "/"))
    con.execute(sql)


def build_tables(csv_path: Path = RAW_CSV, sql_dir: Path = SQL_DIR) -> duckdb.DuckDBPyConnection:
    con = connect(csv_path)
    for name in SCRIPTS:
        run_script(con, sql_dir / name, csv_path)
    return con


def fetch(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()