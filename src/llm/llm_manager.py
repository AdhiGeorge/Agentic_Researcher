
import logging
import os
import json
import time
import random
from typing import Dict, List, Any, Optional

from src.config.system_config import LLMConfig

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manager for interacting with various LLM providers.
    This is a simplified implementation for the demo.
    In a real system, this would use the OpenAI/Azure APIs.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        logger.info(f"Initialized LLM manager with model: {config.model_name}")
        
        # In a real implementation, this would initialize the LLM client
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using the LLM
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate (overrides config)
            
        Returns:
            Generated text
        """
        # In a real implementation, this would call the LLM API
        # For this demo, we'll simulate responses
        
        logger.info(f"Generating text with prompt: {prompt[:50]}...")
        
        # Simulate API delay
        time.sleep(random.uniform(0.5, 1.5))
        
        # Get actual max tokens
        max_tokens = max_tokens or self.config.max_tokens
        
        # Generate a simulated response based on the prompt
        response = self._simulate_response(prompt, max_tokens)
        
        logger.info(f"Generated {len(response)} characters")
        return response
    
    def _simulate_response(self, prompt: str, max_tokens: int) -> str:
        """Simulate an LLM response for demo purposes"""
        # Check if the prompt is asking for a numeric rating
        if "rate" in prompt.lower() and "scale of 0.0 to 1.0" in prompt.lower():
            # Return a random score as requested
            return str(random.uniform(0.65, 0.95))
        
        # Check if it's a code generation prompt
        if "generate python code" in prompt.lower() or "generate code" in prompt.lower():
            return self._simulate_code_response(prompt)
        
        # Check if it's asking for a title
        if "provide a short, descriptive title" in prompt.lower():
            return self._simulate_title_response(prompt)
        
        # Check if it's asking for insights
        if "extract 5 key insights" in prompt.lower():
            return self._simulate_insights_response(prompt)
        
        # Default response for research queries
        return self._simulate_research_response(prompt)
    
    def _simulate_research_response(self, prompt: str) -> str:
        """Simulate a research response"""
        # Extract key terms from the prompt
        terms = [word.lower() for word in prompt.split() 
                if len(word) > 4 and word.isalpha() and word.lower() not in 
                ["based", "research", "query", "context", "information", "provide", "comprehensive"]]
        
        # Use a few random terms to create a response
        selected_terms = random.sample(terms, min(5, len(terms))) if terms else ["research", "topic", "analysis"]
        
        paragraphs = []
        
        # Introduction
        intro = f"The research on {' and '.join(selected_terms)} reveals several important findings. "
        intro += f"When examining {selected_terms[0] if selected_terms else 'this topic'}, it's important to consider multiple perspectives. "
        paragraphs.append(intro)
        
        # Main content
        for i in range(3):
            para = ""
            para += f"Analysis of {random.choice(selected_terms) if selected_terms else 'the subject'} indicates "
            para += f"significant relationships with {random.choice(selected_terms) if selected_terms else 'related concepts'}. "
            para += f"Research by Smith et al. (2022) demonstrates that {random.choice(selected_terms) if selected_terms else 'this area'} "
            para += f"has implications for {random.choice(selected_terms) if selected_terms else 'practical applications'}. "
            para += f"Furthermore, recent studies highlight the importance of understanding {random.choice(selected_terms) if selected_terms else 'these concepts'} "
            para += f"in greater depth, particularly in relation to {random.choice(selected_terms) if selected_terms else 'broader contexts'}."
            paragraphs.append(para)
        
        # Conclusion
        conclusion = f"In conclusion, the research on {' and '.join(selected_terms[:2]) if selected_terms else 'this topic'} "
        conclusion += f"provides valuable insights for future work. Further research is needed to explore additional aspects "
        conclusion += f"and implications of {random.choice(selected_terms) if selected_terms else 'these findings'}."
        paragraphs.append(conclusion)
        
        return "\n\n".join(paragraphs)
    
    def _simulate_code_response(self, prompt: str) -> str:
        """Simulate a code generation response"""
        # Extract key terms from the prompt
        terms = [word.lower() for word in prompt.split() 
                if len(word) > 4 and word.isalpha() and word.lower() not in 
                ["python", "code", "generate", "implementation", "function", "class"]]
        
        # Use a few random terms to create variable and function names
        selected_terms = random.sample(terms, min(3, len(terms))) if terms else ["data", "process", "analyze"]
        
        # Determine what kind of code to generate
        if "visualization" in prompt.lower():
            return self._simulate_visualization_code(selected_terms)
        elif "utility" in prompt.lower() or "helper" in prompt.lower():
            return self._simulate_utility_code(selected_terms)
        else:
            return self._simulate_implementation_code(selected_terms)
    
    def _simulate_implementation_code(self, terms: List[str]) -> str:
        """Simulate a main implementation code response"""
        term1 = terms[0] if terms else "data"
        term2 = terms[1] if len(terms) > 1 else "process"
        
        return f"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class {term1.capitalize()}Processor:
    \"\"\"
    A class for processing {term1} and extracting useful information.
    \"\"\"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        \"\"\"
        Initialize the {term1} processor with optional configuration.
        
        Args:
            config: Configuration parameters
        \"\"\"
        self.config = config or {{
            "threshold": 0.75,
            "max_iterations": 100,
            "use_cache": True
        }}
    
    def process_{term1}(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"
        Process the input {term1} and extract insights.
        
        Args:
            input_data: DataFrame containing the {term1} to process
            
        Returns:
            Dictionary with processed results
        \"\"\"
        results = {{}}
        
        # Validate input data
        if input_data.empty:
            return {{"status": "error", "message": "Input data is empty"}}
        
        # Check for required columns
        required_columns = ["id", "value", "timestamp"]
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            return {{"status": "error", "message": f"Missing required columns: {{missing_columns}}"}}
        
        # Process the data
        processed_data = self._preprocess(input_data)
        features = self._extract_features(processed_data)
        clusters = self._cluster_data(features)
        
        # Compile results
        results["status"] = "success"
        results["n_samples"] = len(input_data)
        results["n_clusters"] = len(clusters)
        results["features"] = features.describe().to_dict()
        results["clusters"] = clusters
        
        return results
    
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Preprocess the input data\"\"\"
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Normalize values
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
        return data
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Extract features from the data\"\"\"
        features = pd.DataFrame()
        
        # Basic features
        features["value_mean"] = data.groupby("id")["value"].mean()
        features["value_std"] = data.groupby("id")["value"].std().fillna(0)
        features["count"] = data.groupby("id").size()
        
        # Time-based features
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        features["days_range"] = data.groupby("id").apply(
            lambda x: (x["timestamp"].max() - x["timestamp"].min()).days
        )
        
        return features
    
    def _cluster_data(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        \"\"\"Cluster the data based on features\"\"\"
        # Simple clustering based on feature values
        clusters = {{}}
        
        # High value cluster
        high_value = features[features["value_mean"] > 0.7].index.tolist()
        clusters["high_value"] = high_value
        
        # High variance cluster
        high_variance = features[features["value_std"] > 0.3].index.tolist()
        clusters["high_variance"] = high_variance
        
        # High activity cluster
        high_activity = features[features["count"] > features["count"].median()].index.tolist()
        clusters["high_activity"] = high_activity
        
        return clusters


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({{
        "id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        "value": [0.5, 0.6, 0.7, 0.3, 0.35, 0.8, 0.85, 0.9, 0.95],
        "timestamp": [
            "2023-01-01", "2023-01-02", "2023-01-03",
            "2023-01-01", "2023-01-03",
            "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"
        ]
    }})
    
    # Initialize processor and process data
    processor = {term1.capitalize()}Processor()
    results = processor.process_{term1}(sample_data)
    
    # Print results
    print(f"Processed {{results['n_samples']}} samples")
    print(f"Found {{results['n_clusters']}} clusters")
    print("Clusters:")
    for cluster_name, cluster_ids in results["clusters"].items():
        print(f"  {{cluster_name}}: {{cluster_ids}}")
"""
    
    def _simulate_visualization_code(self, terms: List[str]) -> str:
        """Simulate a visualization code response"""
        term1 = terms[0] if terms else "data"
        
        return f"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create sample data related to {term1}
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
data = pd.DataFrame({{
    "date": dates,
    "{term1}_value": np.cumsum(np.random.randn(100)) + 10,
    "category": np.random.choice(["A", "B", "C"], size=100),
    "volume": np.random.randint(100, 1000, size=100)
}})

# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"{term1.capitalize()} Analysis Dashboard", fontsize=16)

# Plot 1: Time series of {term1}_value
axs[0, 0].plot(data["date"], data["{term1}_value"], color="blue", linewidth=2)
axs[0, 0].set_title(f"{term1.capitalize()} Value Over Time")
axs[0, 0].set_xlabel("Date")
axs[0, 0].set_ylabel(f"{term1.capitalize()} Value")
axs[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Distribution of {term1}_value
sns.histplot(data["{term1}_value"], kde=True, ax=axs[0, 1])
axs[0, 1].set_title(f"Distribution of {term1.capitalize()} Values")
axs[0, 1].set_xlabel(f"{term1.capitalize()} Value")
axs[0, 1].set_ylabel("Frequency")

# Plot 3: Box plot by category
sns.boxplot(x="category", y="{term1}_value", data=data, ax=axs[1, 0])
axs[1, 0].set_title(f"{term1.capitalize()} Value by Category")
axs[1, 0].set_xlabel("Category")
axs[1, 0].set_ylabel(f"{term1.capitalize()} Value")

# Plot 4: Scatter plot with size representing volume
scatter = axs[1, 1].scatter(
    data.index,
    data["{term1}_value"],
    c=data["category"].astype("category").cat.codes,
    s=data["volume"] / 30,
    alpha=0.6,
    cmap="viridis"
)
axs[1, 1].set_title(f"{term1.capitalize()} Value vs Index (size=volume)")
axs[1, 1].set_xlabel("Index")
axs[1, 1].set_ylabel(f"{term1.capitalize()} Value")

# Add a legend for categories
categories = data["category"].unique()
legend1 = axs[1, 1].legend(
    scatter.legend_elements()[0],
    categories,
    title="Category",
    loc="upper left"
)
axs[1, 1].add_artist(legend1)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig(f"{term1}_analysis_dashboard.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()

# Generate additional insights
print(f"Average {term1}_value: {{data['{term1}_value'].mean():.2f}}")
print(f"Standard deviation: {{data['{term1}_value'].std():.2f}}")
print(f"Min value: {{data['{term1}_value'].min():.2f}}, Max value: {{data['{term1}_value'].max():.2f}}")
print("\\nCategory statistics:")
print(data.groupby("category")["{term1}_value"].agg(["mean", "std", "min", "max"]))
"""
    
    def _simulate_utility_code(self, terms: List[str]) -> str:
        """Simulate utility code response"""
        term1 = terms[0] if terms else "data"
        term2 = terms[1] if len(terms) > 1 else "process"
        
        return f"""
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

def load_{term1}(file_path: str) -> Union[pd.DataFrame, Dict[str, Any]]:
    \"\"\"
    Load {term1} from a file. Supports CSV, Excel, JSON, and pickle formats.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded data as DataFrame or dictionary
        
    Raises:
        ValueError: If the file format is not supported
    \"\"\"
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {{file_path}}")
    
    # Load based on extension
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    elif ext == ".json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {{ext}}")

def clean_{term1}(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    \"\"\"
    Clean the {term1} DataFrame by handling missing values, outliers, and duplicates.
    
    Args:
        data: Input DataFrame to clean
        **kwargs: Additional options including:
            - drop_columns: List of columns to drop
            - fill_method: Method for filling NA values ('mean', 'median', 'mode', 'zero')
            - outlier_threshold: Z-score threshold for outlier detection
            - remove_duplicates: Whether to remove duplicate rows
    
    Returns:
        Cleaned DataFrame
    \"\"\"
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Drop specified columns
    drop_columns = kwargs.get("drop_columns", [])
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")
    
    # Handle missing values
    fill_method = kwargs.get("fill_method", "mean")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            if fill_method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_method == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_method == "zero":
                df[col] = df[col].fillna(0)
    
    # Handle categorical missing values
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna("unknown")
    
    # Handle outliers
    outlier_threshold = kwargs.get("outlier_threshold", 3.0)
    for col in df.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df.loc[z_scores > outlier_threshold, col] = np.nan
        df[col] = df[col].fillna(df[col].median())
    
    # Remove duplicates
    if kwargs.get("remove_duplicates", True):
        df = df.drop_duplicates()
    
    return df

def extract_features_from_{term1}(data: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    \"\"\"
    Extract useful features from the {term1} DataFrame.
    
    Args:
        data: Input DataFrame
        date_col: Name of the date column, if any
    
    Returns:
        DataFrame with extracted features
    \"\"\"
    # Make a copy
    df = data.copy()
    
    # Process date column if specified
    if date_col and date_col in df.columns:
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        
        # Extract date components
        df[f"{{date_col}}_year"] = df[date_col].dt.year
        df[f"{{date_col}}_month"] = df[date_col].dt.month
        df[f"{{date_col}}_day"] = df[date_col].dt.day
        df[f"{{date_col}}_dayofweek"] = df[date_col].dt.dayofweek
        df[f"{{date_col}}_quarter"] = df[date_col].dt.quarter
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_col])
    
    # One-hot encode categorical variables
    cat_columns = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_columns:
        # Only one-hot encode if number of unique values is reasonable
        if df[col].nunique() < 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    
    # Generate interaction features for numerical columns
    num_columns = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5 to avoid explosion
    for i, col1 in enumerate(num_columns):
        for col2 in num_columns[i+1:]:
            df[f"{{col1}}_times_{{col2}}"] = df[col1] * df[col2]
            df[f"{{col1}}_plus_{{col2}}"] = df[col1] + df[col2]
            df[f"{{col1}}_minus_{{col2}}"] = df[col1] - df[col2]
    
    return df

def evaluate_{term2}_results(
    true_values: np.ndarray, 
    predicted_values: np.ndarray
) -> Dict[str, float]:
    \"\"\"
    Evaluate the performance of a {term2} model.
    
    Args:
        true_values: Array of true/actual values
        predicted_values: Array of predicted values
    
    Returns:
        Dictionary with evaluation metrics
    \"\"\"
    # Ensure inputs are numpy arrays
    y_true = np.array(true_values)
    y_pred = np.array(predicted_values)
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared calculation
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
    
    return {{
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "mape": mape
    }}

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({{
        "id": range(100),
        "value": np.random.randn(100) * 10 + 50,
        "category": np.random.choice(["A", "B", "C"], size=100),
        "date": pd.date_range(start="2023-01-01", periods=100)
    }})
    
    # Save to CSV for demonstration
    sample_data.to_csv("sample_{term1}.csv", index=False)
    
    # Load data
    loaded_data = load_{term1}("sample_{term1}.csv")
    print(f"Loaded {{loaded_data.shape[0]}} rows and {{loaded_data.shape[1]}} columns")
    
    # Clean data
    cleaned_data = clean_{term1}(loaded_data, outlier_threshold=2.5)
    print(f"After cleaning: {{cleaned_data.shape[0]}} rows")
    
    # Extract features
    features_df = extract_features_from_{term1}(cleaned_data, date_col="date")
    print(f"Features dataframe has {{features_df.shape[1]}} columns")
    
    # Demo evaluation
    true_vals = np.random.randn(100) * 5 + 20
    pred_vals = true_vals + np.random.randn(100) * 2  # Add some noise
    metrics = evaluate_{term2}_results(true_vals, pred_vals)
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {{metric}}: {{value:.4f}}")
"""
    
    def _simulate_title_response(self, prompt: str) -> str:
        """Simulate a title response"""
        # Extract the query or code snippet from the prompt
        code_match = prompt.find("Code:")
        query_match = prompt.find("Research Query:")
        
        if code_match != -1:
            # This is a code title prompt
            code_snippet = prompt[code_match + 5:code_match + 200]
            
            # Look for keywords in the code
            code_keywords = []
            if "import matplotlib" in code_snippet or "import seaborn" in code_snippet:
                code_keywords.append("Data Visualization")
            if "class" in code_snippet:
                code_keywords.append("Class Implementation")
            if "def load_" in code_snippet or "def clean_" in code_snippet:
                code_keywords.append("Data Processing Utilities")
            if "def evaluate_" in code_snippet:
                code_keywords.append("Performance Evaluation")
            
            if code_keywords:
                return random.choice(code_keywords)
            else:
                return "Python Implementation for Analysis"
        
        elif query_match != -1:
            # This is a research title prompt
            query = prompt[query_match + 15:query_match + 100]
            words = [w for w in query.split() if len(w) > 3 and w.isalpha()]
            
            if words:
                selected_words = random.sample(words, min(3, len(words)))
                return "Analysis of " + " and ".join(selected_words).title()
            else:
                return "Comprehensive Research Analysis"
        
        else:
            # Default title
            return "Analysis and Implementation"
    
    def _simulate_insights_response(self, prompt: str) -> str:
        """Simulate a list of insights response"""
        # Extract key terms from the prompt
        terms = [word.lower() for word in prompt.split() 
                if len(word) > 4 and word.isalpha() and word.lower() not in 
                ["based", "research", "query", "context", "information", "extract", "insight", "finding"]]
        
        # Use a few random terms to create insights
        selected_terms = random.sample(terms, min(5, len(terms))) if terms else ["research", "data", "analysis", "trend", "result"]
        
        insights = []
        
        # Generate 5 insights
        insights.append(f"Recent studies show a significant correlation between {selected_terms[0] if selected_terms else 'key factors'} and overall performance metrics.")
        insights.append(f"The analysis reveals that {selected_terms[1] if len(selected_terms) > 1 else 'primary elements'} has increased by approximately 27% in the last five years.")
        insights.append(f"Contrary to previous assumptions, {selected_terms[2] if len(selected_terms) > 2 else 'the subject'} demonstrates a non-linear relationship with dependent variables.")
        insights.append(f"Researchers have identified three distinct patterns in {selected_terms[3] if len(selected_terms) > 3 else 'the data'}, suggesting multiple underlying mechanisms.")
        insights.append(f"The integration of {selected_terms[4] if len(selected_terms) > 4 else 'advanced techniques'} offers promising avenues for future applications and developments.")
        
        return "\n".join(insights)
