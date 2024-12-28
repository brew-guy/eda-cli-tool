from .csv_reader import CSVReader
from .google_sheets_reader import GoogleSheetsReader
from .xlsx_reader import XLSXReader
from .parquet_reader import ParquetReader
from .json_reader import JSONReader  # Add this import

def get_data_reader(source: str):
    if source.startswith('gs://'):
        return GoogleSheetsReader()
    elif source.endswith('.csv'):
        return CSVReader()
    elif source.endswith('.tsv'):
        return CSVReader(delimiter='\t')
    elif source.endswith('.xlsx'):
        return XLSXReader()
    elif source.endswith('.parquet'):
        return ParquetReader()
    elif source.endswith('.json'):
        return JSONReader()
    else:
        raise ValueError("Unsupported file type")
