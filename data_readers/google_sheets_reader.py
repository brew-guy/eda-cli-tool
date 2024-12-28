import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from pathlib import Path
import pandas as pd
from .base_reader import BaseReader

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_PATH = Path.home() / '.eda' / 'token.pickle'
CREDENTIALS_DIR = Path.home() / '.eda'

class GoogleSheetsReader(BaseReader):
    def get_google_credentials(self) -> Credentials:
        creds = None
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        if TOKEN_PATH.exists():
            with open(TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_DIR / 'client_secrets.json'),
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, 'wb') as token:
                pickle.dump(creds, token)
        return creds

    def read_data(self, source: str, sheet_index: int = 0) -> pd.DataFrame:
        sheet_id = source.replace('gs://', '')
        credentials = self.get_google_credentials()
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheets()[sheet_index]
        return self.get_sheet_as_df(worksheet)

    def get_sheet_as_df(self, worksheet):
        data = worksheet.get_all_values()
        if not data:
            return pd.DataFrame()
        headers = data[0]
        rows = data[1:]
        converted_rows = [[self.infer_type(value) for value in row] for row in rows]
        df = pd.DataFrame(converted_rows, columns=headers)
        object_columns = df.select_dtypes(include=['object']).columns
        df[object_columns] = df[object_columns].astype('string')
        return df

    def infer_type(self, value):
        if value == '':
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                try:
                    return pd.to_datetime(value)
                except (ValueError, TypeError):
                    return str(value)
