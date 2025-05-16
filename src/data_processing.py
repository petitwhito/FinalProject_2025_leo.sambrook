# src/data_processing.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_kuairec_data(data_path="../data/KuaiRec/data/"):
    """
    Load all KuaiRec dataset files.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing KuaiRec data files
        
    Returns:
    --------
    dict
        Dictionary containing all loaded dataframes
    """
    print("Loading KuaiRec datasets...")
    
    # Load interaction data
    print("Loading interaction matrices...")
    small_matrix = pd.read_csv(os.path.join(data_path, "small_matrix.csv"), low_memory=True)
    big_matrix = pd.read_csv(os.path.join(data_path, "big_matrix.csv"), low_memory=True)
    
    # Load side information
    print("Loading social network data...")
    social_network = pd.read_csv(os.path.join(data_path, "social_network.csv"), low_memory=True)
    # Convert string representation of list to actual list
    social_network["friend_list"] = social_network["friend_list"].apply(eval)
    
    print("Loading item categories...")
    item_categories = pd.read_csv(os.path.join(data_path, "item_categories.csv"), low_memory=True)
    # Convert string representation of list to actual list
    item_categories["feat"] = item_categories["feat"].apply(eval)
    
    print("Loading user features...")
    user_features = pd.read_csv(os.path.join(data_path, "user_features.csv"), low_memory=True)
    
    print("Loading item daily features...")
    item_daily_features = pd.read_csv(os.path.join(data_path, "item_daily_features.csv"), low_memory=True)
    
    print("All data loaded successfully!")
    
    # Return all dataframes in a dictionary
    return {
        "small_matrix": small_matrix,
        "big_matrix": big_matrix,
        "social_network": social_network,
        "item_categories": item_categories,
        "user_features": user_features,
        "item_daily_features": item_daily_features,
    }

def check_data_quality(df, name="DataFrame"):
    """
    Check data quality issues like missing values, duplicates, etc.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    name : str
        Name of the DataFrame for reporting purposes
        
    Returns:
    --------
    dict
        Dictionary containing data quality metrics
    """
    print(f"\n--- Data Quality Check for {name} ---")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    print("\nMissing Values:")
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Duplicates - handle unhashable types like lists
    try:
        duplicates = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    except TypeError:
        print("\nCould not check for duplicates due to unhashable types (like lists) in the DataFrame")
        hashable_cols = [col for col in df.columns if not df[col].apply(lambda x: isinstance(x, (list, dict))).any()]
        if hashable_cols:
            duplicates = df.duplicated(subset=hashable_cols).sum()
            print(f"Duplicate rows (considering only hashable columns {hashable_cols}): {duplicates} ({duplicates/len(df)*100:.2f}%)")
        else:
            print("No hashable columns found to check for duplicates")
        duplicates = 0 
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Basic statistics - handle columns with lists
    print("\nBasic Statistics:")
    try:
        print(df.describe(include='all').T)
    except TypeError:
        # For DataFrames with unhashable types, describe only numeric and object columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print("Numeric columns:")
            print(df[numeric_cols].describe().T)
        
        object_cols = df.select_dtypes(include=['object']).columns
        object_cols = [col for col in object_cols if not df[col].apply(lambda x: isinstance(x, (list, dict))).any()]
        if len(object_cols) > 0:
            print("\nObject columns (excluding lists/dicts):")
            print(df[object_cols].describe().T)
        
        # Report on list columns separately
        list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]
        if list_cols:
            print("\nList columns:")
            for col in list_cols:
                list_lengths = df[col].apply(len)
                print(f"\n{col} statistics:")
                print(f"  Min length: {list_lengths.min()}")
                print(f"  Max length: {list_lengths.max()}")
                print(f"  Mean length: {list_lengths.mean():.2f}")
                print(f"  Median length: {list_lengths.median()}")
    
    return {
        "shape": df.shape,
        "missing": missing_df,
        "duplicates": duplicates,
        "dtypes": df.dtypes
    }

def preprocess_interaction_data(df, timestamp_to_datetime=True):
    """
    Preprocess interaction data (small_matrix or big_matrix).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Interaction DataFrame to preprocess
    timestamp_to_datetime : bool
        Whether to convert timestamp to datetime
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert timestamp to datetime if needed
    if timestamp_to_datetime and 'timestamp' in processed_df.columns:
        processed_df['datetime'] = pd.to_datetime(processed_df['timestamp'], unit='s')
        processed_df['date'] = processed_df['datetime'].dt.date
        processed_df['hour'] = processed_df['datetime'].dt.hour
        processed_df['day_of_week'] = processed_df['datetime'].dt.dayofweek
    
    # Handle missing values if any
    if processed_df.isnull().sum().sum() > 0:
        # For numerical columns, fill with median
        num_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    return processed_df

def create_train_test_split(interaction_df, test_size=0.2, by_user=True, random_state=42):
    """
    Split interaction data into training and testing sets.
    
    Parameters:
    -----------
    interaction_df : pandas.DataFrame
        Interaction DataFrame to split
    test_size : float
        Proportion of data to use for testing
    by_user : bool
        If True, split interactions for each user separately
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Training and testing DataFrames
    """
    np.random.seed(random_state)
    
    if by_user:
        # Split interactions for each user
        train_dfs = []
        test_dfs = []
        
        for user_id, user_data in interaction_df.groupby('user_id'):
            # Shuffle the user's interactions
            user_data = user_data.sample(frac=1, random_state=random_state)
            
            # Split into train and test
            n_test = max(1, int(len(user_data) * test_size))
            user_train = user_data.iloc[:-n_test]
            user_test = user_data.iloc[-n_test:]
            
            train_dfs.append(user_train)
            test_dfs.append(user_test)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
    else:
        train_df = interaction_df.sample(frac=1-test_size, random_state=random_state)
        test_df = interaction_df.drop(train_df.index)
    
    print(f"Training set: {len(train_df)} interactions")
    print(f"Testing set: {len(test_df)} interactions")
    
    return train_df, test_df

def visualize_data_distribution(df, column, title=None, bins=30, kde=True):
    """
    Visualize the distribution of a column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name to visualize
    title : str
        Plot title
    bins : int
        Number of bins for histogram
    kde : bool
        Whether to show kernel density estimate
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=kde)
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()