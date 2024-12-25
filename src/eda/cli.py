import click
from eda.core import analyze_data, get_sheets_list
import shutil
from pathlib import Path
from pyfiglet import Figlet
import gspread
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def print_banner():
    """Print a cool ASCII art banner with rainbow colors."""
    f = Figlet(font='slant')
    # Split the banner into lines
    banner_lines = f.renderText('EDA Tool').rstrip().split('\n')
    # Create a rainbow color scheme
    colors = ['bright_red', 'yellow', 'bright_green', 'bright_blue', 'bright_magenta']
    
    # Print each line with its color
    for line, color in zip(banner_lines, colors):
        click.echo(click.style(line, fg=color, bold=True))
    
    # Add a subtle tagline
    click.echo(click.style("\nExploratory Data Analysis Tool", fg='white', dim=True))

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """EDA - Command line tool for exploratory data analysis."""
    # Always show banner
    print_banner()
    
    # If no command is given, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@main.group()
def auth():
    """Manage Google Sheets authentication."""
    pass

@auth.command()
@click.argument('client_secrets_file', type=click.Path(exists=True))
def setup(client_secrets_file):
    """
    Set up Google Sheets authentication with a client secrets file.
    
    Get your client_secrets.json file from Google Cloud Console:
    1. Go to https://console.cloud.google.com
    2. Create a project or select existing project
    3. Enable Google Sheets API
    4. Go to Credentials
    5. Create OAuth 2.0 Client ID (Desktop application)
    6. Download the client secrets file
    """
    credentials_dir = Path.home() / '.eda'
    credentials_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the client secrets file
    shutil.copy2(client_secrets_file, credentials_dir / 'client_secrets.json')
    click.echo("Authentication setup complete. You'll be prompted to authenticate in your browser when needed.")

def select_sheet(source: str) -> int:
    """
    Interactive sheet selector for Google Sheets.
    
    Args:
        source: Google Sheets ID
        
    Returns:
        Selected sheet index
    """
    sheet_id = source.replace('gs://', '')
    credentials = get_google_credentials()
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open_by_key(sheet_id)
    
    sheets = get_sheets_list(spreadsheet)
    
    click.echo("\nAvailable sheets:")
    for idx, title in sheets:
        click.echo(f"{idx}: {title}")
    
    while True:
        sheet_idx = click.prompt(
            "\nSelect sheet number",
            type=int,
            default=0
        )
        if 0 <= sheet_idx < len(sheets):
            click.echo(f"\nSelected sheet: {sheets[sheet_idx][1]}")
            return sheet_idx
        click.echo("Invalid selection. Please try again.")

@main.command()
@click.argument('source')
@click.option('--output', '-o', help='Output file for the analysis')
@click.option('--sheet', '-s', type=int, help='Sheet index (for Google Sheets)')
@click.option('--llm', is_flag=True, help='Include LLM-based analysis using Ollama')
@click.option('--model', default='llama3.2', help='Ollama model to use (default: llama3.2)')
@click.option('--viz', is_flag=True, help='Generate interactive visualizations')
@click.option('--prompt', help='Specific prompt template to use (default: auto-detect)')
def analyze(source, output, sheet, llm, model, viz, prompt):
    """
    Analyze a data file and generate summary statistics.
    
    SOURCE can be either:
    - Path to a local CSV file
    - Google Sheets ID (prefix with 'gs://')
    
    Example:
        eda analyze data.csv
        eda analyze data.csv --llm --viz
        eda analyze data.csv --llm --prompt timeseries
        eda analyze gs://1234567890abcdef --llm --model codellama --viz
    """
    if source.startswith('gs://') and sheet is None:
        sheet = select_sheet(source)
    
    result = analyze_data(source, sheet or 0, llm=llm, model=model, viz=viz, prompt_type=prompt)
    if output:
        with open(output, 'w') as f:
            f.write(result)
    else:
        click.echo(result)

def get_google_credentials():
    """
    Get or refresh Google OAuth2 credentials.
    
    Returns:
        google.oauth2.credentials.Credentials: The OAuth2 credentials
    """
    credentials = None
    token_path = Path.home() / '.eda' / 'token.pickle'
    credentials_path = Path.home() / '.eda' / 'client_secrets.json'

    # Load existing credentials if available
    if token_path.exists():
        with open(token_path, 'rb') as token:
            credentials = pickle.load(token)

    # If credentials are invalid or don't exist, get new ones
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                ['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            credentials = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open(token_path, 'wb') as token:
            pickle.dump(credentials, token)

    return credentials

if __name__ == "__main__":
    main()
