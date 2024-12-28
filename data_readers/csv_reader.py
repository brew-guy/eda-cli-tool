import pandas as pd
from .base_reader.py import BaseReader

class CSVReader(BaseReader):
    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        return pd.read_csv(source)
