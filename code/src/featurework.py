import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """数据质量检查"""
    if df.empty:
        raise ValueError("输入数据为空！")

    # 检查必要列是否存在
    required_cols = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失必要列: {missing_cols}")

    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """更稳健的技术指标计算"""
    df = check_data_quality(df)
    df = df.sort_values(['股票代码', '日期'])

    # 基础特征
    df = df.assign(
        close_open_ratio=np.clip(df['收盘'] / df['开盘'], 0.8, 1.2),
        high_low_ratio=np.clip(df['最高'] / df['最低'], 1.0, 1.2),
        daily_return=df.groupby('股票代码')['收盘'].pct_change()
    )

    # 滚动特征(添加最小周期限制)
    windows = [3, 5, 10]
    for w in windows:
        df[f'ma{w}'] = df.groupby('股票代码')['收盘'].transform(
            lambda x: x.rolling(w, min_periods=2).mean())
        df[f'std{w}'] = df.groupby('股票代码')['收盘'].transform(
            lambda x: x.rolling(w, min_periods=2).std())

    # 目标变量
    df['future_return'] = df.groupby('股票代码')['收盘'].transform(
        lambda x: np.clip(x.shift(-5) / x - 1, -0.5, 0.5))

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """更安全的特征处理"""
    # 日期特征
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期'])
    df['day_of_week'] = df['日期'].dt.dayofweek
    df['month'] = df['日期'].dt.month

    # 填充缺失值
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df.groupby('股票代码')[col].transform(
            lambda x: x.fillna(x.rolling(10, min_periods=1).mean()))

    # 温和的异常值处理
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols, how='all')

    # 股票编码
    df['stock_code'] = df['股票代码'].astype('category').cat.codes
    df['original_stock_code'] = df['股票代码']

    return df


def prepare_train_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = calculate_technical_indicators(df)
        df = process_features(df)
        if df.empty:
            raise ValueError("预处理后数据为空！")
        return df
    except Exception as e:
        raise ValueError(f"数据准备失败: {str(e)}")


def prepare_test_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = calculate_technical_indicators(df)
        df = process_features(df)
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    except Exception as e:
        raise ValueError(f"测试数据准备失败: {str(e)}")