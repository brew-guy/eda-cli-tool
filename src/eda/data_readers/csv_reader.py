import pandas as pd
from .base_reader import BaseReader

class CSVReader(BaseReader):
    def __init__(self, delimiter=','):
        self.delimiter = delimiter

    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        return pd.read_csv(source, delimiter=self.delimiter)
