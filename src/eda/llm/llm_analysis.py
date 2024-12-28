import pandas as pd
import ollama
from rich.console import Console
from pathlib import Path
import yaml

console = Console()

def load_prompt_template(prompt_type: str = 'default') -> str:
    prompt_path = Path(__file__).parent.parent / 'prompts' / f'{prompt_type}.yaml'
    if not prompt_path.exists():
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'default.yaml'
    with open(prompt_path) as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['template']

def detect_data_type(df: pd.DataFrame) -> str:
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
    prompt_template = load_prompt_template(prompt_type)
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
        markdown_text = response['response']
        return markdown_text
    except Exception as e:
        console.print(f"[bold red]Error getting LLM analysis: {str(e)}[/]")
        return ""
