import pandas as pd
from .base_reader import BaseReader

class ParquetReader(BaseReader):
    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        return pd.read_parquet(source)
