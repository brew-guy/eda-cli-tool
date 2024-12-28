import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, Any, Tuple

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    return numeric_cols.corr()

def perform_statistical_tests(df: pd.DataFrame) -> str:
    """Perform basic statistical tests on the data."""
    output_lines = []
    
    # Normality tests for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    for col in numeric_cols:
        stat, p_value = stats.normaltest(df[col].dropna())
        output_lines.append(f"{col} Normality Test:")
        output_lines.append(f"  Statistic: {stat:.4f}")
        output_lines.append(f"  P-value: {p_value:.4f}")
        output_lines.append(f"  Result: {'Normal' if p_value > 0.05 else 'Not normal'}")
    
    return "\n".join(output_lines)

def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, np.ndarray]:
    """Detect outliers using Z-score method."""
    outliers = {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers[col] = df[col][z_scores > threshold]
    
    return outliers
