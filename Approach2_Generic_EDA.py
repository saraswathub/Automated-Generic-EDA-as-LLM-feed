import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# Initialize the transformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def perform_detailed_textual_eda(dataset_path, n_components=2):
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return f"Error loading dataset: {e}"

    # Initialize a description string
    description = ""

    # Basic Information
    description += "--- Basic Information ---\n"
    description += f"Number of Rows: {df.shape[0]}\n"
    description += f"Number of Columns: {df.shape[1]}\n"
    description += "\nColumn Names:\n"
    description += ", ".join(df.columns.tolist()) + "\n\n"

    # Data Types
    description += "--- Data Types ---\n"
    description += df.dtypes.to_string() + "\n\n"

    # Missing Values
    description += "--- Missing Values ---\n"
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Values": missing_values,
        "Percentage": missing_percentage
    })
    description += missing_summary.to_string() + "\n\n"

    # Constant Columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    description += "--- Constant Columns ---\n"
    if constant_columns:
        description += f"The dataset contains {len(constant_columns)} constant column(s): {', '.join(constant_columns)}\n"
    else:
        description += "No constant columns detected.\n\n"

    # Duplicate Rows
    duplicate_rows = df.duplicated().sum()
    description += "--- Duplicate Rows ---\n"
    description += f"Number of duplicate rows: {duplicate_rows}\n\n"

    # Summary Statistics (Numeric Columns)
    description += "--- Summary Statistics (Numeric Columns) ---\n"
    numeric_summary = df.describe().T
    description += numeric_summary.to_string() + "\n\n"

    # Skewness and Kurtosis
    description += "--- Skewness and Kurtosis (Numeric Columns) ---\n"
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        skewness = df[column].skew()
        kurtosis = df[column].kurt()
        description += f"{column}: Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}\n"
    description += "\n"

    # Unique Values Per Column
    description += "--- Unique Values Per Column ---\n"
    unique_values = {col: df[col].nunique() for col in df.columns}
    for col, count in unique_values.items():
        description += f"{col}: {count} unique values\n"
    description += "\n"

    # Value Counts for Categorical Columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    description += "--- Value Counts (Categorical Columns) ---\n"
    for column in categorical_columns:
        description += f"\nColumn: {column}\n"
        description += df[column].value_counts().to_string() + "\n"
    description += "\n"

    # Correlation Analysis
    description += "--- Correlation Analysis (Numeric Columns) ---\n"
    if not numeric_columns.empty:
        correlation_matrix = df[numeric_columns].corr()
        description += "Correlation Matrix:\n"
        description += correlation_matrix.to_string() + "\n\n"

        # Significant Correlations (Above Threshold)
        significant_threshold = 0.7
        significant_correlations = correlation_matrix[(correlation_matrix > significant_threshold) & (correlation_matrix != 1)]
        description += f"Significant correlations (>|{significant_threshold}|):\n"
        description += significant_correlations.to_string() + "\n\n"
    else:
        description += "No numeric columns available for correlation analysis.\n\n"

    # Outlier Detection (Using IQR)
    description += "--- Outlier Detection (Numeric Columns) ---\n"
    outlier_summary = {}
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_summary[column] = outliers

    for column, outliers in outlier_summary.items():
        description += f"{column}: {outliers} outlier(s) detected\n"
    description += "\n"

    # One Hot Encoding for Categorical Columns
    description += "--- One Hot Encoding ---\n"
    if not categorical_columns.empty:
        encoder = OneHotEncoder(sparse=False)
        encoded_df = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
        encoded_df.columns = encoder.get_feature_names_out(categorical_columns)
        df = df.drop(columns=categorical_columns).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        description += "One hot encoding applied to categorical columns.\n\n"
    else:
        description += "No categorical columns available for one hot encoding.\n\n"

    # PCA for Dimensionality Reduction
    description += "--- PCA for Dimensionality Reduction ---\n"
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    description += f"PCA applied with {n_components} components. Explained variance ratio: {pca.explained_variance_ratio_}\n\n"

    description += "Extended EDA Completed."

    return description

def generate_vector_embeddings(sentences):
    embeddings = model.encode(sentences)
    return embeddings


dataset_path = '/Users/ballu_macbookpro/Downloads/titanic.csv'
eda_description = perform_detailed_textual_eda(dataset_path)
print(eda_description)

# Generate vector embeddings for extracted values
extracted_values = eda_description.split('\n')
embeddings = generate_vector_embeddings(extracted_values)
print("\n--- Vector Embeddings ---\n")
for value, embedding in zip(extracted_values, embeddings):
    print(f"Value: {value}\nEmbedding: {embedding}\n")
