# Content-based filtering models for KuaiRec recommender system
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedModel:
    """Content-based filtering model."""
    
    def __init__(self):
        """Initialize content-based model."""
        self.item_profiles = None
        self.item_ids = None
        self.user_profiles = {}
        self.content_columns = None
        self.scaler = StandardScaler()
        
    def fit(self, item_features_df, train_df, content_columns=None, 
            user_col='user_id', item_col='video_id', rating_col='watch_ratio'):
        """
        Train the model on the data.
        
        Parameters:
        -----------
        item_features_df : pandas.DataFrame
            DataFrame with item features
        train_df : pandas.DataFrame
            Training data
        content_columns : list, optional
            List of columns to use as content features
        user_col : str
            Column name for user IDs
        item_col : str
            Column name for item IDs
        rating_col : str
            Column name for ratings
        """
        # If no content columns specified, use all columns that start with 'category_'
        if content_columns is None:
            content_columns = [col for col in item_features_df.columns if col.startswith('category_')]
            
            # Add some additional columns if available
            additional_cols = ['watch_ratio_mean', 'engagement_score']
            for col in additional_cols:
                if col in item_features_df.columns:
                    content_columns.append(col)
        
        self.content_columns = content_columns
        
        # Extract item IDs and features
        self.item_ids = item_features_df['video_id'].values
        
        # Create item profiles
        item_profiles = item_features_df[content_columns].values
        
        # Normalize the features
        self.item_profiles = self.scaler.fit_transform(item_profiles)
        
        # Create a mapping from item ID to profile index
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        # Create user profiles as weighted average of item profiles
        for user_id, group in train_df.groupby(user_col):
            # Get the items this user has interacted with
            user_items = []
            user_ratings = []
            
            for _, row in group.iterrows():
                item_id = row[item_col]
                rating = row[rating_col]
                
                if item_id in item_id_to_idx:
                    user_items.append(item_id_to_idx[item_id])
                    user_ratings.append(rating)
            
            if len(user_items) > 0:
                # Calculate weighted average of item profiles
                user_ratings = np.array(user_ratings)
                user_ratings = user_ratings / user_ratings.sum()  # Normalize weights
                
                user_profile = np.zeros(self.item_profiles.shape[1])
                for i, item_idx in enumerate(user_items):
                    user_profile += user_ratings[i] * self.item_profiles[item_idx]
                
                self.user_profiles[user_id] = user_profile
        
        return self
    
    def save(self, filepath):
        """Save the model to a file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'item_profiles': self.item_profiles,
                'item_ids': self.item_ids,
                'user_profiles': self.user_profiles,
                'content_columns': self.content_columns,
                'scaler': self.scaler
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.item_profiles = data['item_profiles']
        model.item_ids = data['item_ids']
        model.user_profiles = data['user_profiles']
        model.content_columns = data['content_columns']
        model.scaler = data['scaler']
        return model 