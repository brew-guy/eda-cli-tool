import pandas as pd
import rich_click as click
from eda.core import analyze_data, get_sheets_list
from eda.data_readers.google_sheets_reader import GoogleSheetsReader
import shutil
from pathlib import Path
from pyfiglet import Figlet
import gspread
from rich.console import Console
from rich.markdown import Markdown

# Use Rich markup
click.rich_click.USE_RICH_MARKUP = True

console = Console()

def print_banner():
    """Print a cool ASCII art banner with a rainbow color effect."""
    f = Figlet(font='slant')
    banner_text = f.renderText('EDA Tool').rstrip()
    
    # Apply rainbow color effect
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    colored_banner = ""
    for i, line in enumerate(banner_text.split('\n')):
        color = colors[i % len(colors)]
        colored_banner += f"[{color}]{line}[/]\n"
    
    console.print(colored_banner, markup=True)
    console.print("[bold]Exploratory Data Analysis Tool[/]", markup=True)

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

def authenticate_google_sheets():
    """Authenticate and cache Google Sheets credentials."""
    reader = GoogleSheetsReader()
    credentials = reader.get_google_credentials()
    gc = gspread.authorize(credentials)
    print("Google Sheets authentication successful.")

def select_sheet(source: str) -> int:
    """
    Interactive sheet selector for Google Sheets and Excel files.
    
    Args:
        source: Google Sheets ID or Excel file path
        
    Returns:
        Selected sheet index
    """
    if source.startswith('gs://'):
        sheet_id = source.replace('gs://', '')
        reader = GoogleSheetsReader()
        credentials = reader.get_google_credentials()
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_key(sheet_id)
        
        sheets = get_sheets_list(spreadsheet)
    else:
        xls = pd.ExcelFile(source)
        sheets = [(i, sheet) for i, sheet in enumerate(xls.sheet_names)]
    
    console.print("\n[bold green]Available sheets:[/]", markup=True)
    for idx, title in sheets:
        console.print(f"[cyan]{idx}[/]: {title}", markup=True)
    
    while True:
        sheet_idx = console.input("\n[bold yellow]Select sheet number[/]: ")
        try:
            sheet_idx = int(sheet_idx)
            if 0 <= sheet_idx < len(sheets):
                console.print(f"\n[bold green]Selected sheet:[/] {sheets[sheet_idx][1]}", markup=True)
                return sheet_idx
        except ValueError:
            pass
        console.print("[bold red]Invalid selection. Please try again.[/]", markup=True)

@main.command()
@click.argument('source')
@click.option('--output', '-o', help='Output file for the analysis')
@click.option('--sheet', '-s', type=int, help='Sheet index (for Google Sheets and Excel files)')
@click.option('--llm', is_flag=True, help='Include LLM-based analysis using Ollama')
@click.option('--model', default='llama3.2', help='Ollama model to use (default: llama3.2)')
@click.option('--viz', is_flag=True, help='Generate interactive visualizations')
@click.option('--prompt', help='Specific prompt template to use (default: auto-detect)')
def analyze(source, output, sheet, llm, model, viz, prompt):
    """
    Analyze a data file and generate summary statistics.
    
    SOURCE can be either:
    - Path to a local CSV or Excel file
    - Google Sheets ID (prefix with 'gs://')
    
    Example:
        eda analyze data.csv
        eda analyze data.xlsx --sheet 1
        eda analyze data.csv --llm --viz
        eda analyze data.csv --llm --prompt timeseries
        eda analyze gs://1234567890abcdef --llm --model codellama --viz
    """
    if (source.startswith('gs://') or source.endswith('.xlsx')) and sheet is None:
        sheet = select_sheet(source)
    
    result, llm_output = analyze_data(source, sheet or 0, llm=llm, model=model, viz=viz, prompt_type=prompt)
    if output:
        with open(output, 'w') as f:
            f.write(result)
            if llm_output:
                f.write("\n\nLLM Analysis:\n")
                f.write(llm_output)
    else:
        console.print(result, markup=True)
        if llm_output:
            console.print(Markdown(llm_output))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EDA Tool")
    parser.add_argument(
        'auth', 
        help="Authenticate Google Sheets",
        action='store_true'
    )
    args = parser.parse_args()
    
    if args.auth:
        authenticate_google_sheets()
    main()
