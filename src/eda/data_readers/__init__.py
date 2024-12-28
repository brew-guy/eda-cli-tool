from .csv_reader import CSVReader
from .google_sheets_reader import GoogleSheetsReader
from .xlsx_reader import XLSXReader
from .parquet_reader import ParquetReader

def get_data_reader(source: str):
    if source.startswith('gs://'):
        return GoogleSheetsReader()
    elif source.endswith('.csv'):
        return CSVReader()
    elif source.endswith('.xlsx'):
        return XLSXReader()
    elif source.endswith('.parquet'):
        return ParquetReader()
    else:
        raise ValueError("Unsupported file type")
