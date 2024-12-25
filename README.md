# EDA Tool

A command line tool for Exploratory Data Analysis.

## Installation

This project uses Poetry for dependency management. To get started:

1. Install Poetry (if you haven't already):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/jhun/eda.git
   cd eda
   ```

3. Install dependencies:

   ```bash
   poetry install
   ```

4. Activate the virtual environment:

   ```bash
   poetry shell
   ```

## Usage

The tool supports both local CSV files and Google Sheets:

### Basic Analysis of Local CSV File

Analyze a local CSV file and display basic statistics:

```bash
poetry run eda analyze data.csv
```

### Analyze Google Sheet (Interactive Sheet Selection)

Analyze a Google Sheet by providing the sheet ID. The tool will prompt you to select a specific sheet if there are multiple sheets:

```bash
poetry run eda analyze gs://your_sheet_id
```

### Analyze Specific Sheet in Google Sheet

Analyze a specific sheet in a Google Sheet by providing the sheet index:

```bash
poetry run eda analyze gs://your_sheet_id --sheet 0
```

### Analysis with LLM Insights Using Ollama

Include LLM-based analysis using the default Ollama model:

```bash
poetry run eda analyze data.csv --llm
```

Specify a different Ollama model for the analysis:

```bash
poetry run eda analyze data.csv --llm --model codellama
```

### Generate Interactive Visualizations

Generate interactive visualizations for the dataset:

```bash
poetry run eda analyze data.csv --viz
```

### Combine LLM Analysis and Visualizations

Combine LLM-based analysis and interactive visualizations:

```bash
poetry run eda analyze data.csv --llm --viz
```

### Use a Specific Prompt Template

Use a specific prompt template for the LLM analysis. Available templates are `default`, `numeric`, `categorical`, and `timeseries`:

```bash
poetry run eda analyze data.csv --prompt timeseries
```

### Save Analysis to File

Save the analysis output to a file:

```bash
poetry run eda analyze data.csv -o analysis.txt
```

## Development

### Development Tools

The project includes several development tools:

- **pytest** for testing:

  ```bash
  poetry run pytest
  ```

- **black** for code formatting:

  ```bash
  poetry run black src/ tests/
  ```

- **isort** for import sorting:
  ```bash
  poetry run isort src/ tests/
  ```

### Running the CLI

For development, you can run the CLI directly through poetry:

```bash
poetry run eda analyze data.csv
```

## License

[Choose a license]

## Google Sheets Setup

To use Google Sheets:

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Enable the Google Sheets API for your project
4. Go to the Credentials page
5. Create OAuth 2.0 Client ID credentials
   - Application type: Desktop application
   - Download the client secrets file
6. Set up authentication with:
   ```bash
   eda auth setup path/to/client_secrets.json
   ```
7. When you first access a Google Sheet, your browser will open and ask you to authenticate

The tool will cache your credentials in `~/.eda/token.pickle` for future use.

Usage example:

```bash
# Get the sheet ID from the Google Sheets URL
# https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit
poetry run eda analyze gs://YOUR_SHEET_ID
```
