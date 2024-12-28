import pandas as pd
from .base_reader import BaseReader

class JSONReader(BaseReader):
    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        with open(source, 'r') as file:
            return pd.read_json(file, lines=True)
