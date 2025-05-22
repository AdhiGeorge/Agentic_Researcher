"""
Data Visualization for Agentic Researcher

This module provides visualization tools for analyzing tabular data
and generating interactive charts for research results.
"""

import os
import io
import logging
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Handles data visualization for research analysis
    
    Supports:
    - Basic statistical analysis
    - Common chart types (bar, line, scatter, etc.)
    - Chart export to various formats
    """
    
    def __init__(self, theme: str = "default"):
        """
        Initialize data visualizer
        
        Args:
            theme: Visual theme for charts ('default', 'dark', 'light', 'colorblind')
        """
        self.theme = theme
        self._setup_theme()
        
    def _setup_theme(self):
        """Configure visualization theme"""
        if self.theme == "dark":
            plt.style.use('dark_background')
        elif self.theme == "light":
            plt.style.use('seaborn-v0_8-whitegrid')
        elif self.theme == "colorblind":
            plt.style.use('seaborn-v0_8-colorblind')
        else:
            # Default theme
            plt.style.use('seaborn-v0_8')
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of statistics
        """
        try:
            # Basic info
            info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Numerical stats
            num_cols = df.select_dtypes(include=['number']).columns
            num_stats = {}
            
            if len(num_cols) > 0:
                # Summary statistics
                num_stats = {
                    "numerical": {
                        "count": df[num_cols].count().to_dict(),
                        "mean": df[num_cols].mean().to_dict(),
                        "std": df[num_cols].std().to_dict(),
                        "min": df[num_cols].min().to_dict(),
                        "25%": df[num_cols].quantile(0.25).to_dict(),
                        "50%": df[num_cols].quantile(0.50).to_dict(),
                        "75%": df[num_cols].quantile(0.75).to_dict(),
                        "max": df[num_cols].max().to_dict()
                    }
                }
            
            # Categorical stats
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            cat_stats = {}
            
            if len(cat_cols) > 0:
                cat_stats = {
                    "categorical": {
                        col: {
                            "unique_count": df[col].nunique(),
                            "top_values": df[col].value_counts().head(5).to_dict()
                        } for col in cat_cols
                    }
                }
            
            # Missing values
            missing = {
                "missing": {
                    col: int(df[col].isna().sum()) for col in df.columns
                }
            }
            
            # Combine all stats
            stats = {**info, **num_stats, **cat_stats, **missing}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating summary stats: {str(e)}")
            return {"error": str(e)}
    
    def plot_histogram(self, 
                      df: pd.DataFrame, 
                      column: str,
                      bins: int = 20,
                      title: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 6)) -> Dict[str, Any]:
        """
        Create histogram for a numerical column
        
        Args:
            df: Input DataFrame
            column: Column name to plot
            bins: Number of bins
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Dictionary with plot metadata and base64-encoded image
        """
        try:
            if column not in df.columns:
                return {"error": f"Column '{column}' not found in DataFrame"}
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                return {"error": f"Column '{column}' is not numeric"}
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Generate histogram
            sns.histplot(df[column].dropna(), bins=bins, kde=True)
            
            # Add labels and title
            plt.xlabel(column)
            plt.ylabel("Frequency")
            if title:
                plt.title(title)
            else:
                plt.title(f"Distribution of {column}")
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode image
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                "plot_type": "histogram",
                "column": column,
                "image": img_str,
                "format": "png",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return {"error": str(e)}
    
    def plot_bar(self, 
                df: pd.DataFrame, 
                column: str,
                limit: int = 10,
                title: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 6)) -> Dict[str, Any]:
        """
        Create bar plot for a categorical column
        
        Args:
            df: Input DataFrame
            column: Column name to plot
            limit: Maximum number of categories to show
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Dictionary with plot metadata and base64-encoded image
        """
        try:
            if column not in df.columns:
                return {"error": f"Column '{column}' not found in DataFrame"}
            
            # Get value counts and take top N
            value_counts = df[column].value_counts().head(limit)
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Generate bar plot
            sns.barplot(x=value_counts.index, y=value_counts.values)
            
            # Add labels and title
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            if title:
                plt.title(title)
            else:
                plt.title(f"Top {limit} values of {column}")
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode image
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                "plot_type": "bar",
                "column": column,
                "image": img_str,
                "format": "png",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating bar plot: {str(e)}")
            return {"error": str(e)}
    
    def plot_scatter(self, 
                    df: pd.DataFrame, 
                    x_column: str,
                    y_column: str,
                    color_column: Optional[str] = None,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 6)) -> Dict[str, Any]:
        """
        Create scatter plot for two numerical columns
        
        Args:
            df: Input DataFrame
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            color_column: Optional column for color coding points
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Dictionary with plot metadata and base64-encoded image
        """
        try:
            if x_column not in df.columns:
                return {"error": f"Column '{x_column}' not found in DataFrame"}
            
            if y_column not in df.columns:
                return {"error": f"Column '{y_column}' not found in DataFrame"}
            
            if color_column and color_column not in df.columns:
                return {"error": f"Column '{color_column}' not found in DataFrame"}
            
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                return {"error": f"Column '{x_column}' is not numeric"}
            
            if not pd.api.types.is_numeric_dtype(df[y_column]):
                return {"error": f"Column '{y_column}' is not numeric"}
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Generate scatter plot
            if color_column:
                scatter = sns.scatterplot(x=x_column, y=y_column, hue=color_column, data=df)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(x=x_column, y=y_column, data=df)
            
            # Add labels and title
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            if title:
                plt.title(title)
            else:
                plt.title(f"{y_column} vs {x_column}")
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode image
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                "plot_type": "scatter",
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column,
                "image": img_str,
                "format": "png",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return {"error": str(e)}
    
    def plot_line(self, 
                 df: pd.DataFrame, 
                 x_column: str,
                 y_columns: Union[str, List[str]],
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (10, 6)) -> Dict[str, Any]:
        """
        Create line plot for one or more y-columns against an x-column
        
        Args:
            df: Input DataFrame
            x_column: Column name for x-axis
            y_columns: Column name(s) for y-axis (single string or list)
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Dictionary with plot metadata and base64-encoded image
        """
        try:
            if x_column not in df.columns:
                return {"error": f"Column '{x_column}' not found in DataFrame"}
            
            # Convert single y_column to list
            if isinstance(y_columns, str):
                y_columns = [y_columns]
            
            # Check if all y_columns exist
            for col in y_columns:
                if col not in df.columns:
                    return {"error": f"Column '{col}' not found in DataFrame"}
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Plot each y-column
            for col in y_columns:
                plt.plot(df[x_column], df[col], label=col)
            
            # Add labels and title
            plt.xlabel(x_column)
            plt.ylabel("Value")
            plt.legend()
            if title:
                plt.title(title)
            else:
                if len(y_columns) == 1:
                    plt.title(f"{y_columns[0]} over {x_column}")
                else:
                    plt.title(f"Values over {x_column}")
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode image
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                "plot_type": "line",
                "x_column": x_column,
                "y_columns": y_columns,
                "image": img_str,
                "format": "png",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating line plot: {str(e)}")
            return {"error": str(e)}
    
    def plot_correlation_matrix(self, 
                               df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               title: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8)) -> Dict[str, Any]:
        """
        Create correlation matrix heatmap
        
        Args:
            df: Input DataFrame
            columns: List of columns to include (defaults to all numeric columns)
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Dictionary with plot metadata and base64-encoded image
        """
        try:
            # If columns not specified, use all numeric columns
            if columns is None:
                columns = df.select_dtypes(include=['number']).columns.tolist()
            else:
                # Verify all columns exist and are numeric
                for col in columns:
                    if col not in df.columns:
                        return {"error": f"Column '{col}' not found in DataFrame"}
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        return {"error": f"Column '{col}' is not numeric"}
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Generate heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                       square=True, linewidths=0.5)
            
            # Add title
            if title:
                plt.title(title)
            else:
                plt.title("Correlation Matrix")
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode image
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                "plot_type": "correlation_matrix",
                "columns": columns,
                "image": img_str,
                "format": "png",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            return {"error": str(e)}
    
    def generate_visualization_report(self, 
                                     df: pd.DataFrame,
                                     output_dir: str = "./exports",
                                     filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive visualization report for a DataFrame
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save report
            filename: HTML filename for report
            
        Returns:
            Dictionary with report metadata and file path
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_visualization_report_{timestamp}.html"
            
            # Create full file path
            file_path = os.path.join(output_dir, filename)
            
            # Generate summary statistics
            stats = self.generate_summary_stats(df)
            
            # Create plots
            plots = {}
            
            # Generate histograms for numeric columns (up to 10)
            numeric_cols = df.select_dtypes(include=['number']).columns[:10]
            for col in numeric_cols:
                plots[f"histogram_{col}"] = self.plot_histogram(df, col)
            
            # Generate bar plots for categorical columns (up to 5)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
            for col in cat_cols:
                plots[f"bar_{col}"] = self.plot_bar(df, col)
            
            # Generate correlation matrix if at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plots["correlation_matrix"] = self.plot_correlation_matrix(df)
            
            # Generate HTML report
            html_content = self._generate_html_report(df, stats, plots)
            
            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"Visualization report generated: {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "statistics": stats,
                "plots": {k: v["plot_type"] for k, v in plots.items() if "error" not in v},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_html_report(self, 
                             df: pd.DataFrame, 
                             stats: Dict[str, Any], 
                             plots: Dict[str, Any]) -> str:
        """
        Generate HTML report from statistics and plots
        
        Args:
            df: Input DataFrame
            stats: Dictionary of statistics
            plots: Dictionary of plots
            
        Returns:
            HTML content as string
        """
        # HTML header
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Visualization Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                    margin-top: 24px;
                    margin-bottom: 16px;
                }}
                h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: .3em; }}
                h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: .3em; }}
                pre {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 16px;
                    overflow: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 16px;
                }}
                table, th, td {{
                    border: 1px solid #dfe2e5;
                }}
                th, td {{
                    padding: 12px 16px;
                    text-align: left;
                }}
                th {{
                    background-color: #f6f8fa;
                }}
                .plot-container {{
                    margin-bottom: 30px;
                }}
                .plots-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                    gap: 20px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>Data Visualization Report</h1>
            <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            
            <h2>Dataset Overview</h2>
            <ul>
                <li><strong>Rows:</strong> {stats.get('shape', [0, 0])[0]}</li>
                <li><strong>Columns:</strong> {stats.get('shape', [0, 0])[1]}</li>
            </ul>
            
            <h3>Data Sample</h3>
            <pre>{df.head(5).to_html()}</pre>
            
            <h2>Summary Statistics</h2>
        """
        
        # Add numerical statistics table if available
        if "numerical" in stats:
            html += "<h3>Numerical Columns</h3>"
            html += "<pre>"
            
            # Create a DataFrame from the numerical stats
            num_stats = stats["numerical"]
            num_stats_df = pd.DataFrame(num_stats)
            
            html += num_stats_df.to_html()
            html += "</pre>"
        
        # Add missing values information
        if "missing" in stats:
            missing = stats["missing"]
            total_missing = sum(missing.values())
            
            html += f"<h3>Missing Values</h3>"
            html += f"<p>Total missing values: {total_missing}</p>"
            
            if total_missing > 0:
                html += "<ul>"
                for col, count in missing.items():
                    if count > 0:
                        percentage = (count / len(df)) * 100
                        html += f"<li><strong>{col}:</strong> {count} ({percentage:.2f}%)</li>"
                html += "</ul>"
        
        # Add categorical statistics if available
        if "categorical" in stats:
            html += "<h3>Categorical Columns</h3>"
            
            for col, cat_stats in stats["categorical"].items():
                html += f"<h4>{col}</h4>"
                html += f"<p>Unique values: {cat_stats['unique_count']}</p>"
                
                if cat_stats['top_values']:
                    html += "<p>Top values:</p>"
                    html += "<ul>"
                    for val, count in cat_stats['top_values'].items():
                        percentage = (count / len(df)) * 100
                        html += f"<li>{val}: {count} ({percentage:.2f}%)</li>"
                    html += "</ul>"
        
        # Add plots
        html += "<h2>Visualizations</h2>"
        html += "<div class='plots-grid'>"
        
        for name, plot in plots.items():
            if "error" not in plot and "image" in plot:
                plot_title = name.replace("_", " ").title()
                html += f"""
                <div class='plot-container'>
                    <h3>{plot_title}</h3>
                    <img src="data:image/png;base64,{plot['image']}" alt="{name}">
                </div>
                """
        
        html += "</div>"
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        return html
