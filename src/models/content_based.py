import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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
        """Train model on item features and user interactions."""
        # Find which columns to use for content features
        if content_columns is None:
            content_columns = [col for col in item_features_df.columns if col.startswith('category_')]
            
            # Add engagement metrics if available
            additional_cols = ['watch_ratio_mean', 'engagement_score']
            for col in additional_cols:
                if col in item_features_df.columns:
                    content_columns.append(col)
        
        self.content_columns = content_columns
        
        # Get item IDs and features
        self.item_ids = item_features_df['video_id'].values
        
        # Create item profiles
        item_profiles = item_features_df[content_columns].values
        
        # Normalize features
        self.item_profiles = self.scaler.fit_transform(item_profiles)
        
        # Create lookup table for items
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        # Build user profiles from their interactions
        for user_id, group in train_df.groupby(user_col):
            # Track items and ratings
            user_items = []
            user_ratings = []
            
            for _, row in group.iterrows():
                item_id = row[item_col]
                rating = row[rating_col]
                
                if item_id in item_id_to_idx:
                    user_items.append(item_id_to_idx[item_id])
                    user_ratings.append(rating)
            
            if len(user_items) > 0:
                # Create weighted average of item profiles
                user_ratings = np.array(user_ratings)
                user_ratings = user_ratings / user_ratings.sum()
                
                user_profile = np.zeros(self.item_profiles.shape[1])
                for i, item_idx in enumerate(user_items):
                    user_profile += user_ratings[i] * self.item_profiles[item_idx]
                
                self.user_profiles[user_id] = user_profile
        
        return self
    
    def recommend(self, user_id, n=10, exclude_items=None):
        """Get top-N recommendations for a user."""
        if user_id not in self.user_profiles:
            return []
        
        # Get user profile
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity with all items
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.item_profiles)[0]
        
        # Remove excluded items
        if exclude_items:
            for item_id in exclude_items:
                try:
                    content_idx = np.where(self.item_ids == item_id)[0][0]
                    similarities[content_idx] = -np.inf
                except:
                    pass
        
        # Find top items
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        # Return recommendations with scores
        return [(self.item_ids[idx], similarities[idx]) for idx in top_indices]
    
    def save(self, filepath):
        """Save model to disk."""
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
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.item_profiles = data['item_profiles']
        model.item_ids = data['item_ids']
        model.user_profiles = data['user_profiles']
        model.content_columns = data['content_columns']
        model.scaler = data['scaler']
        return model 