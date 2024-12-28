import pandas as pd

class BaseReader:
    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method")
