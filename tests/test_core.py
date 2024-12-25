import pytest
from eda.core import (
    analyze_data, 
    read_data, 
    format_markdown, 
    detect_data_type,
    load_prompt_template,
    get_llm_analysis
)
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from google.oauth2.credentials import Credentials
import click

def test_analyze_data_local(tmp_path):
    # Create a test CSV file
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)
    
    # Test the analyze function
    result = analyze_data(str(test_file))
    assert "Shape:" in result
    assert "A" in result
    assert "B" in result

def test_analyze_data_with_llm(tmp_path):
    df = pd.DataFrame({'A': [1, 2, 3]})
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)
    
    with patch('ollama.generate') as mock_generate:
        mock_generate.return_value = {'response': '# Test\nAnalysis content'}
        result = analyze_data(str(test_file), llm=True)
        assert "Test" in result
        assert "Analysis content" in result

def test_format_markdown():
    markdown = """# Header
**bold text**
*italic text*
`code`
- bullet point"""
    
    result = format_markdown(markdown)
    assert click.style("Header", fg='bright_cyan', bold=True) in result
    assert "•" in result  # Check bullet point conversion

def test_detect_data_type():
    # Test numeric data
    df_numeric = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4.0, 5.0, 6.0]
    })
    assert detect_data_type(df_numeric) == 'numeric'
    
    # Test categorical data
    df_cat = pd.DataFrame({
        'A': ['x', 'y', 'z'],
        'B': ['a', 'b', 'c']
    })
    assert detect_data_type(df_cat) == 'categorical'
    
    # Test timeseries data
    df_time = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'value': [1, 2]
    })
    assert detect_data_type(df_time) == 'timeseries'

def test_load_prompt_template():
    template = load_prompt_template('default')
    assert '{rows}' in template
    assert '{columns}' in template
    
    # Test fallback to default template
    template = load_prompt_template('nonexistent')
    assert '{rows}' in template

@patch('webbrowser.open')
def test_analyze_data_with_viz(mock_browser, tmp_path):
    df = pd.DataFrame({'A': [1, 2, 3]})
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)
    
    result = analyze_data(str(test_file), viz=True)
    assert "Visualizations opened in your browser" in result
    mock_browser.assert_called_once()
