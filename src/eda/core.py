"""Core functionality for EDA tool."""
import pandas as pd
import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from pathlib import Path
from typing import List, Tuple
import ollama
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import tempfile
import yaml
import rich_click as click
from rich.console import Console

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.MAX_WIDTH = 100
click.rich_click.SHOW_ARGUMENTS = True

# Create console instance for rich output
console = Console()

# If modifying these scopes, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_PATH = Path.home() / '.eda' / 'token.pickle'
CREDENTIALS_DIR = Path.home() / '.eda'

def get_google_credentials() -> Credentials:
    """
    Get Google credentials using OAuth2 with browser authentication.
    Caches the credentials in ~/.eda/token.pickle for future use.
    
    Returns:
        Credentials object
    """
    creds = None
    
    # Ensure credentials directory exists
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load cached credentials if they exist
    if TOKEN_PATH.exists():
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_DIR / 'client_secrets.json'),
                SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def get_sheets_list(spreadsheet: gspread.Spreadsheet) -> List[Tuple[int, str]]:
    """
    Get list of available sheets in the spreadsheet.
    
    Args:
        spreadsheet: Google Spreadsheet object
    
    Returns:
        List of tuples containing (index, sheet_name)
    """
    return [(i, sheet.title) for i, sheet in enumerate(spreadsheet.worksheets())]

def read_data(source: str, sheet_index: int = 0) -> pd.DataFrame:
    """
    Read data from either a local file or Google Sheets.
    
    Args:
        source: Path to local file or Google Sheets ID
        sheet_index: Index of the sheet to read (for Google Sheets only)
    
    Returns:
        pandas DataFrame
    """
    if source.startswith('gs://'):
        # Google Sheets format: gs://sheet_id
        sheet_id = source.replace('gs://', '')
        try:
            credentials = get_google_credentials()
            gc = gspread.authorize(credentials)
            spreadsheet = gc.open_by_key(sheet_id)
            
            # Get the selected worksheet
            worksheet = spreadsheet.worksheets()[sheet_index]
            return get_sheet_as_df(worksheet)
        except FileNotFoundError:
            raise Exception(
                "Client secrets file not found. Please run 'eda auth setup' first."
            )
    else:
        # Local file
        return pd.read_csv(source)

def format_section(title: str, content: str) -> str:
    """Format a section with title and content."""
    return f"\n[bold green]{title}[/]\n{'='*len(title)}\n{content}"

def analyze_data(source, sheet_index=0, llm=False, model='llama3.2', viz=False, prompt_type=None):
    """
    Analyze the data from the given source and return the analysis result and LLM output.
    
    Args:
        source (str): Path to the data file or Google Sheets ID.
        sheet_index (int): Index of the sheet to analyze (for Google Sheets).
        llm (bool): Whether to include LLM-based analysis.
        model (str): The LLM model to use.
        viz (bool): Whether to generate interactive visualizations.
        prompt_type (str): Specific prompt template to use.
    
    Returns:
        Tuple[str, str]: The analysis result and LLM output.
    """
    try:
        df = read_data(source, sheet_index)
        
        # Dataset shape with color
        dataset_overview = format_section("Dataset Overview", f"Shape: [yellow]{df.shape}[/]")
        
        # Column information section
        column_info = format_section("Column Information", df.dtypes.to_string())
        
        # Missing values with highlighting
        missing_values_content = ""
        for col, count in df.isnull().sum().items():
            color = "red" if count > 0 else "green"
            missing_values_content += f"\n{col}: [{color}]{count}[/]"
        missing_values = format_section("Missing Values", missing_values_content)
        
        # Summary statistics
        summary_stats = format_section("Summary Statistics", df.describe(include='all').to_string())
        
        output = [dataset_overview, column_info, missing_values, summary_stats]
        
        if viz:
            fig = create_visualizations(df)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                fig.write_html(f.name)
                webbrowser.open(f'file://{f.name}')
                output.append("\n[magenta]Visualizations opened in your browser.[/]")
        
        llm_output = None
        if llm:
            llm_section = format_section("LLM Analysis", "")
            output.append(llm_section)
            data_type = prompt_type or detect_data_type(df)
            llm_output = get_llm_analysis(df, model, prompt_type=data_type)
        
        return "\n".join(output), llm_output
        
    except Exception as e:
        return f"[bold red]Error analyzing file: {str(e)}[/]", ""

def load_prompt_template(prompt_type: str = 'default') -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / 'prompts' / f'{prompt_type}.yaml'
    if not prompt_path.exists():
        prompt_path = Path(__file__).parent / 'prompts' / 'default.yaml'
    
    with open(prompt_path) as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['template']

def detect_data_type(df: pd.DataFrame) -> str:
    """Detect the primary type of data in the DataFrame."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    
    if 'date' in df.columns or 'timestamp' in df.columns:
        return 'timeseries'
    elif len(numeric_cols) > len(cat_cols) * 2:
        return 'numeric'
    elif len(cat_cols) > len(numeric_cols) * 2:
        return 'categorical'
    return 'default'

def get_llm_analysis(df: pd.DataFrame, model: str, prompt_type: str) -> str:
    """Get LLM-based analysis of the dataset using Ollama."""
    prompt_template = load_prompt_template(prompt_type)
    
    # Format statistics to be more concise
    stats_summary = df.describe(include='all').to_string()
    
    context = prompt_template.format(
        rows=df.shape[0],
        columns=df.shape[1],
        dtypes=df.dtypes.to_string(),
        stats=stats_summary
    )
    
    try:
        response = ollama.generate(
            model=model,
            prompt=context,
            stream=False
        )
        
        # Get the markdown response
        markdown_text = response['response']
        return markdown_text  # Return the markdown text

    except Exception as e:
        console.print(f"[bold red]Error getting LLM analysis: {str(e)}[/]")
        return ""

def infer_type(value):
    """Infer the type of a value from Google Sheets."""
    if value == '':
        return None
    try:
        # Try to convert to int first
        int_val = int(value)
        return int_val
    except ValueError:
        try:
            # Then try float
            float_val = float(value)
            return float_val
        except ValueError:
            try:
                # Then try datetime
                return pd.to_datetime(value)
            except (ValueError, TypeError):
                # Convert to pandas string type
                return str(value)  # We'll convert the whole column later

def get_sheet_as_df(worksheet):
    """Convert Google Sheet to DataFrame with proper type inference."""
    data = worksheet.get_all_values()
    if not data:
        return pd.DataFrame()
    
    headers = data[0]
    rows = data[1:]
    
    # Convert each value using type inference
    converted_rows = []
    for row in rows:
        converted_row = [infer_type(value) for value in row]
        converted_rows.append(converted_row)
    
    df = pd.DataFrame(converted_rows, columns=headers)
    
    # Convert object columns to string dtype
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype('string')
    
    return df

def create_visualizations(df: pd.DataFrame) -> go.Figure:
    """Create a dashboard of visualizations for the dataset."""
    # Calculate number of numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    
    # Create subplot grid based on number of columns
    n_cat = len(cat_cols)
    
    fig = make_subplots(
        rows=2 + (n_cat > 0),  # Add row if we have categorical columns
        cols=2,
        subplot_titles=(
            "Correlation Heatmap", "Distribution Overview",
            "Missing Values", "Time Series" if 'date' in df.columns else "Scatter Matrix",
            "Category Distributions" if n_cat > 0 else None
        )
    )
    
    # 1. Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr, x=corr.columns, y=corr.columns),
            row=1, col=1
        )
    
    # 2. Distribution overview (box plots)
    for i, col in enumerate(numeric_cols):
        fig.add_trace(
            go.Box(y=df[col], name=col),
            row=1, col=2
        )
    
    # 3. Missing values visualization
    missing = df.isnull().sum()
    fig.add_trace(
        go.Bar(x=missing.index, y=missing.values, name="Missing Values"),
        row=2, col=1
    )
    
    # 4. Time series or scatter plot
    if 'date' in df.columns:
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            fig.add_trace(
                go.Scatter(x=df['date'], y=df[col], name=col),
                row=2, col=2
            )
    elif len(numeric_cols) >= 2:
        fig.add_trace(
            go.Scatter(
                x=df[numeric_cols[0]], 
                y=df[numeric_cols[1]], 
                mode='markers'
            ),
            row=2, col=2
        )
    
    # 5. Category distributions (if categorical columns exist)
    if n_cat > 0:
        for i, col in enumerate(cat_cols[:3]):  # Limit to first 3 categorical columns
            value_counts = df[col].value_counts()
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                row=3, col=1 + (i > 1)
            )
    
    # Update layout
    fig.update_layout(
        height=300 * (2 + (n_cat > 0)),
        width=1000,
        title_text="Dataset Visualization Dashboard",
        showlegend=True
    )
    
    return fig
