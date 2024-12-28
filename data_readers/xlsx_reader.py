import pandas as pd
from .base_reader import BaseReader

class XLSXReader(BaseReader):
    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        return pd.read_excel(source, sheet_name=sheet_index)
