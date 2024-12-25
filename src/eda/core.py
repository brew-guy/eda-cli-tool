"""Core functionality for EDA tool."""
import pandas as pd
import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
from pathlib import Path
from typing import List, Tuple
import ollama
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import tempfile
import yaml
import click
import re

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
    return (
        click.style(f"\n{title}", fg="bright_blue", bold=True) +
        click.style("\n" + "="*len(title), fg="bright_blue") +
        "\n" + content
    )

def analyze_data(source: str, sheet_index: int = 0, llm: bool = False, model: str = None, viz: bool = False, prompt_type: str = None) -> str:
    """
    Analyze a data file and return basic statistics with optional LLM analysis and visualizations.
    """
    try:
        df = read_data(source, sheet_index)
        summary = []
        
        # Dataset shape with color
        summary.append(click.style("Dataset Overview", fg="green", bold=True))
        summary.append(click.style("===============", fg="green"))
        summary.append(f"Shape: {click.style(str(df.shape), fg='bright_yellow')}")
        
        # Column information section
        summary.append(format_section("Column Information", df.dtypes.to_string()))
        
        # Missing values with highlighting
        missing_data = df.isnull().sum()
        missing_formatted = []
        for col, count in missing_data.items():
            if count > 0:
                missing_formatted.append(f"{col}: {click.style(str(count), fg='bright_red')}")
            else:
                missing_formatted.append(f"{col}: {click.style('0', fg='bright_green')}")
        summary.append(format_section("Missing Values", "\n".join(missing_formatted)))
        
        # Summary statistics
        summary.append(format_section("Summary Statistics", df.describe(include='all').to_string()))
        
        # Print the statistical analysis immediately
        print("\n".join(summary))
        output = []
        
        if viz:
            print(click.style("\nGenerating visualizations...", fg="bright_magenta", bold=True), flush=True)
            fig = create_visualizations(df)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                fig.write_html(f.name)
                webbrowser.open(f'file://{f.name}')
                # Only print the message, don't add to output
                print(click.style("\nVisualizations opened in your browser.", fg="bright_magenta"))
        
        if llm:
            print(click.style("\nGenerating LLM Analysis...", fg="bright_cyan", bold=True), flush=True)
            # Use specified prompt type or auto-detect
            data_type = prompt_type or detect_data_type(df)
            llm_analysis = get_llm_analysis(df, model, prompt_type=data_type)
            output.append(llm_analysis)
        
        return "\n".join(output)
        
    except Exception as e:
        error_msg = click.style(f"Error analyzing file: {str(e)}", fg="bright_red", bold=True)
        print(error_msg)
        return error_msg

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

def format_markdown(text: str) -> str:
    """Format markdown text for CLI display."""
    # Replace headers (##, ###, etc.)
    for i in range(6, 0, -1):
        pattern = f"{'#' * i} (.*)"
        text = re.sub(pattern, lambda m: click.style(m.group(1), fg='bright_cyan', bold=True), text)
    
    # Replace bold (**text**) - handle both inline and multiline
    text = re.sub(r'\*\*(.*?)\*\*', lambda m: click.style(m.group(1), bold=True), text, flags=re.DOTALL)
    
    # Replace italic (*text*) - handle both inline and multiline
    text = re.sub(r'\*(.*?)\*', lambda m: click.style(m.group(1), italic=True), text, flags=re.DOTALL)
    
    # Replace inline code (`text`)
    text = re.sub(r'`(.*?)`', lambda m: click.style(m.group(1), fg='bright_yellow'), text)
    
    # Replace bullet points
    text = re.sub(r'^- ', '• ', text, flags=re.MULTILINE)
    
    # Replace numbered lists (1., 2., etc.)
    text = re.sub(r'^\d+\.\s', lambda m: click.style(m.group(), fg='bright_magenta'), text, flags=re.MULTILINE)
    
    return text

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
        
        # Parse and format the LLM response
        analysis = response['response']
        sections = analysis.split('\n#')
        formatted_sections = []
        
        if not analysis.startswith('#'):
            formatted_sections.append(click.style(sections[0].strip(), fg='bright_white'))
        
        for section in sections[1:] if analysis.startswith('#') else sections[1:]:
            if section.strip():
                parts = section.split('\n', 1)
                if len(parts) == 2:
                    header, content = parts
                    formatted_sections.append(
                        click.style(f"\n# {header}", fg='bright_cyan', bold=True) +
                        click.style(f"\n{content.strip()}", fg='bright_white')
                    )
        
        return "\n".join(formatted_sections)
    except Exception as e:
        return click.style(f"Error getting LLM analysis: {str(e)}", fg='bright_red', bold=True)

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
    n_numeric = len(numeric_cols)
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
