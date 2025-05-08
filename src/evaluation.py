import os
import numpy as np
import pandas as pd
from collections import defaultdict

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate precision@k for a single user.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of relevant (ground truth) item IDs
    k : int
        Number of recommendations to consider
        
    Returns:
    --------
    float
        Precision@k value
    """
    if len(recommended_items) == 0:
        return 0.0
        
    # Only consider the top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
        
    # Count relevant items in recommendations
    hits = len(set(recommended_items) & set(relevant_items))
    
    return hits / len(recommended_items)

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate recall@k for a single user.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of relevant (ground truth) item IDs
    k : int
        Number of recommendations to consider
        
    Returns:
    --------
    float
        Recall@k value
    """
    if len(relevant_items) == 0 or len(recommended_items) == 0:
        return 0.0
        
    # Only consider the top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
        
    # Count relevant items in recommendations
    hits = len(set(recommended_items) & set(relevant_items))
    
    return hits / len(relevant_items)

def average_precision(recommended_items, relevant_items, k=10):
    """
    Calculate average precision for a single user.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of relevant (ground truth) item IDs
    k : int
        Number of recommendations to consider
        
    Returns:
    --------
    float
        Average precision value
    """
    if len(relevant_items) == 0 or len(recommended_items) == 0:
        return 0.0
        
    # Only consider the top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
        
    hits = 0
    sum_precision = 0.0
    
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precision += precision_at_i
            
    if hits == 0:
        return 0.0
        
    return sum_precision / min(len(relevant_items), k)

def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate NDCG@k for a single user.
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs
    relevant_items : list
        List of relevant (ground truth) item IDs
    k : int
        Number of recommendations to consider
        
    Returns:
    --------
    float
        NDCG@k value
    """
    if len(relevant_items) == 0 or len(recommended_items) == 0:
        return 0.0
        
    # Only consider the top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    
    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            # Using binary relevance (1 if item is relevant)
            rel = 1
            dcg += rel / np.log2(i + 2)  # i+2 because i starts from 0
    
    # Ideal DCG calculation
    ideal_dcg = 0.0
    for i in range(min(len(relevant_items), k)):
        ideal_dcg += 1 / np.log2(i + 2)
    
    if ideal_dcg == 0:
        return 0.0
        
    return dcg / ideal_dcg

def evaluate_recommendations(recommendations_df, test_interactions_df, k_values=[5, 10, 20]):
    """
    Evaluate recommendations using multiple metrics.
    
    Parameters:
    -----------
    recommendations_df : pandas.DataFrame
        DataFrame with columns ['user_id', 'video_id', 'rank', 'score']
    test_interactions_df : pandas.DataFrame
        DataFrame with columns ['user_id', 'video_id', ...]
    k_values : list
        List of k values for evaluation
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Group test interactions by user
    user_interactions = defaultdict(list)
    for _, row in test_interactions_df.iterrows():
        user_interactions[row['user_id']].append(row['video_id'])
    
    # Group recommendations by user
    user_recommendations = defaultdict(list)
    for _, row in recommendations_df.iterrows():
        user_recommendations[row['user_id']].append(row['video_id'])
    
    # Calculate metrics for each k
    results = {}
    
    for k in k_values:
        precision_sum = 0.0
        recall_sum = 0.0
        ap_sum = 0.0
        ndcg_sum = 0.0
        
        user_count = 0
        
        for user_id, relevant_items in user_interactions.items():
            if user_id in user_recommendations:
                recommended_items = user_recommendations[user_id]
                
                precision_sum += precision_at_k(recommended_items, relevant_items, k)
                recall_sum += recall_at_k(recommended_items, relevant_items, k)
                ap_sum += average_precision(recommended_items, relevant_items, k)
                ndcg_sum += ndcg_at_k(recommended_items, relevant_items, k)
                
                user_count += 1
        
        if user_count > 0:
            results[f'precision@{k}'] = precision_sum / user_count
            results[f'recall@{k}'] = recall_sum / user_count
            results[f'map@{k}'] = ap_sum / user_count
            results[f'ndcg@{k}'] = ndcg_sum / user_count
        else:
            results[f'precision@{k}'] = 0.0
            results[f'recall@{k}'] = 0.0
            results[f'map@{k}'] = 0.0
            results[f'ndcg@{k}'] = 0.0
    
    return results

def load_data(processed_dir="../data/processed", results_dir="../results"):
    """
    Load necessary data for evaluation.
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing processed data
    results_dir : str
        Directory containing recommendation results
        
    Returns:
    --------
    tuple
        (recommendations_df, test_interactions_df)
    """
    # Load recommendations
    recommendations_path = os.path.join(results_dir, "recommendations.csv")
    recommendations_df = pd.read_csv(recommendations_path)
    
    # Load test interactions
    test_interactions_path = os.path.join(processed_dir, "test_interactions.csv")
    
    # Sample a small portion of test data if it's too large
    test_interactions_df = pd.read_csv(test_interactions_path, nrows=10000000)
    
    return recommendations_df, test_interactions_df

def evaluate_per_model(recommender, test_interactions_df, users=None, k_values=[5, 10, 20], n=10):
    """
    Evaluate each recommendation model separately.
    
    Parameters:
    -----------
    recommender : KuaiRecRecommender
        Recommender instance
    test_interactions_df : pandas.DataFrame
        DataFrame with test interactions
    users : list, optional
        List of users to evaluate
    k_values : list
        List of k values for evaluation
    n : int
        Number of recommendations to generate
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics for each model
    """
    # Group test interactions by user
    user_interactions = defaultdict(list)
    for _, row in test_interactions_df.iterrows():
        user_interactions[row['user_id']].append(row['video_id'])
    
    # If users not specified, use all users with test interactions
    if users is None:
        users = list(user_interactions.keys())
    
    # Models to evaluate
    models = [
        ('collaborative', {'collaborative': 1.0, 'content': 0.0, 'sequence': 0.0, 'hybrid': 0.0}),
        ('content', {'collaborative': 0.0, 'content': 1.0, 'sequence': 0.0, 'hybrid': 0.0}),
        ('sequence', {'collaborative': 0.0, 'content': 0.0, 'sequence': 1.0, 'hybrid': 0.0}),
        ('hybrid', {'collaborative': 0.0, 'content': 0.0, 'sequence': 0.0, 'hybrid': 1.0}),
        ('combined', {'collaborative': 0.4, 'content': 0.3, 'sequence': 0.2, 'hybrid': 0.1})
    ]
    
    results = {}
    
    for model_name, weights in models:
        print(f"Evaluating {model_name} model...")
        
        # Generate recommendations for each user
        user_recommendations = {}
        for user_id in users:
            if user_id in user_interactions:
                recs = recommender.recommend(user_id, n=n, weights=weights)
                user_recommendations[user_id] = [item_id for item_id, _ in recs]
        
        # Calculate metrics for each k
        model_results = {}
        
        for k in k_values:
            precision_sum = 0.0
            recall_sum = 0.0
            ap_sum = 0.0
            ndcg_sum = 0.0
            
            user_count = 0
            
            for user_id, relevant_items in user_interactions.items():
                if user_id in user_recommendations:
                    recommended_items = user_recommendations[user_id]
                    
                    precision_sum += precision_at_k(recommended_items, relevant_items, k)
                    recall_sum += recall_at_k(recommended_items, relevant_items, k)
                    ap_sum += average_precision(recommended_items, relevant_items, k)
                    ndcg_sum += ndcg_at_k(recommended_items, relevant_items, k)
                    
                    user_count += 1
            
            if user_count > 0:
                model_results[f'precision@{k}'] = precision_sum / user_count
                model_results[f'recall@{k}'] = recall_sum / user_count
                model_results[f'map@{k}'] = ap_sum / user_count
                model_results[f'ndcg@{k}'] = ndcg_sum / user_count
            else:
                model_results[f'precision@{k}'] = 0.0
                model_results[f'recall@{k}'] = 0.0
                model_results[f'map@{k}'] = 0.0
                model_results[f'ndcg@{k}'] = 0.0
        
        results[model_name] = model_results
    
    return results 