import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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