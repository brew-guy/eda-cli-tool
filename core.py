"""Core functionality for EDA tool."""
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import webbrowser
import tempfile
import yaml
from rich.console import Console
from eda.data_readers import get_data_reader
from eda.llm.llm_analysis import get_llm_analysis, detect_data_type
from eda.visualizations.plotly_visualizations import create_visualizations

# Create console instance for rich output
console = Console()

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
        reader = get_data_reader(source)
        df = reader.read_data(source, sheet_index)
        
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
    prompt_path = Path(__file__).parent / 'llm' / 'prompt_templates' / f'{prompt_type}.yaml'
    if not prompt_path.exists():
        prompt_path = Path(__file__).parent / 'llm' / 'prompt_templates' / 'default.yaml'
    
    with open(prompt_path) as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['template']
