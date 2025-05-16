import pandas as pd
import numpy as np
from scipy import sparse
import os

def create_user_features(interaction_df, user_features_df=None):
    """Create user features from interaction history and metadata."""
    # Get basic user stats from interactions
    user_stats = interaction_df.groupby('user_id').agg({
        'watch_ratio': ['mean', 'std', 'min', 'max', 'count'],
        'play_duration': ['mean', 'sum'],
        'video_duration': ['mean']
    })
    
    # Flatten column names
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
    user_stats = user_stats.reset_index()
    
    # Calculate play-to-duration ratio
    user_stats['avg_play_to_duration_ratio'] = user_stats['play_duration_mean'] / user_stats['video_duration_mean']
    
    # Calculate activity level
    user_stats['activity_level'] = pd.qcut(
        user_stats['watch_ratio_count'], 
        q=5, 
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    
    # Merge with user metadata if available
    if user_features_df is not None:
        user_features = user_stats.merge(user_features_df, on='user_id', how='left')
    else:
        user_features = user_stats
    
    return user_features

def create_item_features(interaction_df, item_categories_df=None, item_daily_df=None):
    """Create item features from interaction history and metadata."""
    # Get basic item stats from interactions
    item_stats = interaction_df.groupby('video_id').agg({
        'watch_ratio': ['mean', 'std', 'min', 'max', 'count'],
        'play_duration': ['mean', 'sum'],
    })
    
    # Flatten column names
    item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns.values]
    item_stats = item_stats.reset_index()
    
    # Calculate popularity metrics
    item_stats['popularity'] = pd.qcut(
        item_stats['watch_ratio_count'], 
        q=5, 
        labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )
    
    # Calculate engagement score
    item_stats['engagement_score'] = item_stats['watch_ratio_mean'] * np.log1p(item_stats['watch_ratio_count'])
    
    # Merge with item categories if available
    if item_categories_df is not None:
        # One-hot encode categories
        item_categories = item_categories_df.copy()
        
        # Create category feature columns
        all_categories = set()
        for cats in item_categories['feat']:
            all_categories.update(cats)
            
        for cat in all_categories:
            item_categories[f'category_{cat}'] = item_categories['feat'].apply(lambda x: 1 if cat in x else 0)
        
        # Merge with item stats
        item_features = item_stats.merge(
            item_categories.drop('feat', axis=1), 
            on='video_id', 
            how='left'
        )
    else:
        item_features = item_stats
    
    # Add daily metrics if available
    if item_daily_df is not None:
        daily_agg = item_daily_df.groupby('video_id').agg({
            'play_cnt': 'sum',
            'like_cnt': 'sum',
            'comment_cnt': 'sum',
            'share_cnt': 'sum',
            'play_duration': 'sum'
        }).reset_index()
        
        # Calculate engagement metrics
        daily_agg['like_to_play_ratio'] = daily_agg['like_cnt'] / daily_agg['play_cnt'].clip(lower=1)
        daily_agg['comment_to_play_ratio'] = daily_agg['comment_cnt'] / daily_agg['play_cnt'].clip(lower=1)
        daily_agg['share_to_play_ratio'] = daily_agg['share_cnt'] / daily_agg['play_cnt'].clip(lower=1)
        
        # Merge with item features
        item_features = item_features.merge(daily_agg, on='video_id', how='left', suffixes=('', '_daily'))
        
    return item_features

def create_interaction_features(interaction_df, user_features_df=None, item_features_df=None):
    """Create interaction features combining user and item information."""
    interactions = interaction_df.copy()
    
    # Add temporal features if available
    if 'datetime' in interactions.columns:
        interactions['hour_of_day'] = interactions['datetime'].dt.hour
        interactions['day_of_week'] = interactions['datetime'].dt.dayofweek
        interactions['weekend'] = interactions['day_of_week'].isin([5, 6]).astype(int)
        interactions['time_of_day'] = pd.cut(
            interactions['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
    
    # Add user features if available
    if user_features_df is not None:
        # Select only necessary columns
        user_cols = ['user_id', 'watch_ratio_mean', 'activity_level']
        if 'user_active_degree' in user_features_df.columns:
            user_cols.append('user_active_degree')
        
        interactions = interactions.merge(
            user_features_df[user_cols],
            on='user_id',
            how='left',
            suffixes=('', '_user_avg')
        )
    
    # Add item features if available
    if item_features_df is not None:
        # Select only necessary columns
        item_cols = ['video_id', 'watch_ratio_mean', 'popularity', 'engagement_score']
        
        interactions = interactions.merge(
            item_features_df[item_cols],
            on='video_id',
            how='left',
            suffixes=('', '_item_avg')
        )
    
    # Create derived features
    if 'watch_ratio_mean_item_avg' in interactions.columns and 'watch_ratio' in interactions.columns:
        # Relative watch ratio compared to item average
        interactions['relative_watch_ratio'] = interactions['watch_ratio'] / interactions['watch_ratio_mean_item_avg'].clip(lower=0.1)
    
    if 'watch_ratio_mean_user_avg' in interactions.columns and 'watch_ratio' in interactions.columns:
        # Relative watch ratio compared to user average
        interactions['user_preference'] = interactions['watch_ratio'] / interactions['watch_ratio_mean_user_avg'].clip(lower=0.1)
    
    return interactions

def build_interaction_matrix(interaction_df, rating_col='watch_ratio', normalize=False):
    """Build a user-item interaction matrix from interaction data."""
    # Create user and item indices
    user_indices = {user: i for i, user in enumerate(interaction_df['user_id'].unique())}
    item_indices = {item: i for i, item in enumerate(interaction_df['video_id'].unique())}
    
    # Map user and item IDs to indices
    row_indices = interaction_df['user_id'].map(user_indices).values
    col_indices = interaction_df['video_id'].map(item_indices).values
    
    # Get rating values
    data = interaction_df[rating_col].values
    
    # Build sparse matrix
    n_users = len(user_indices)
    n_items = len(item_indices)
    matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_items))
    
    # Normalize if requested
    if normalize:
        # Normalize by user (row-wise)
        user_means = np.array(matrix.sum(axis=1)).flatten() / np.array((matrix != 0).sum(axis=1)).flatten()
        user_means = np.nan_to_num(user_means)
        for i in range(n_users):
            matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] -= user_means[i]
    
    return matrix, user_indices, item_indices

def extract_social_features(interaction_df, social_network_df):
    """Extract features based on social network information."""
    if social_network_df is None or len(social_network_df) == 0:
        return pd.DataFrame({'user_id': interaction_df['user_id'].unique()})
    
    # Create a dictionary of users and their friends
    user_friends = dict(zip(social_network_df['user_id'], social_network_df['friend_list']))
    
    # Extract interaction data for all users
    user_item_data = interaction_df[['user_id', 'video_id', 'watch_ratio']].copy()
    
    # Create a dictionary to store user-video watch ratios
    video_ratings = {}
    for _, row in user_item_data.iterrows():
        user = row['user_id']
        video = row['video_id']
        rating = row['watch_ratio']
        
        if user not in video_ratings:
            video_ratings[user] = {}
        video_ratings[user][video] = rating
    
    # Calculate social features for each user
    social_features = []
    
    for user_id in interaction_df['user_id'].unique():
        # Initialize features
        features = {
            'user_id': user_id,
            'friend_count': 0,
            'friend_avg_watch_ratio': 0,
            'friend_similarity': 0
        }
        
        # If user has friends, calculate features
        if user_id in user_friends:
            friends = user_friends[user_id]
            features['friend_count'] = len(friends)
            
            # Calculate average watch ratio of friends
            friend_watch_ratios = []
            for friend in friends:
                if friend in video_ratings:
                    friend_watch_ratios.extend(video_ratings[friend].values())
            
            if friend_watch_ratios:
                features['friend_avg_watch_ratio'] = np.mean(friend_watch_ratios)
            
            # Calculate similarity with friends
            if user_id in video_ratings and len(friends) > 0:
                similarities = []
                for friend in friends:
                    if friend in video_ratings:
                        # Find common videos
                        common_videos = set(video_ratings[user_id].keys()) & set(video_ratings[friend].keys())
                        
                        if common_videos:
                            user_ratings = [video_ratings[user_id][v] for v in common_videos]
                            friend_ratings = [video_ratings[friend][v] for v in common_videos]
                            
                            # Calculate cosine similarity
                            similarity = np.dot(user_ratings, friend_ratings) / (
                                np.linalg.norm(user_ratings) * np.linalg.norm(friend_ratings)
                            )
                            similarities.append(similarity)
                
                if similarities:
                    features['friend_similarity'] = np.mean(similarities)
        
        social_features.append(features)
    
    return pd.DataFrame(social_features)

def save_features(features_dict, output_dir):
    """Save features to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in features_dict.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        print(f"Saved {name} with shape {df.shape}") 