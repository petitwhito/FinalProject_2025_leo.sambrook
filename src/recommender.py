# src/recommender.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from scipy import sparse

from src.models.collaborative import ALSModel
from src.models.content_based import ContentBasedModel
from src.models.sequence_aware import SequentialRules
from src.models.hybrid import LightGBMModel

class KuaiRecRecommender:
    """Main recommender system for KuaiRec videos."""
    
    def __init__(self, models_dir="../models", processed_dir="../data/processed"):
        """Set up recommender with model and data directories."""
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        
        # Load mapping files
        with open(os.path.join(processed_dir, "user_indices.pkl"), 'rb') as f:
            self.user_indices = pickle.load(f)
        with open(os.path.join(processed_dir, "item_indices.pkl"), 'rb') as f:
            self.item_indices = pickle.load(f)
            
        # Reverse mappings
        self.index_to_user = {v: k for k, v in self.user_indices.items()}
        self.index_to_item = {v: k for k, v in self.item_indices.items()}
        
        # Load interaction matrix for collaborative filtering
        try:
            self.interaction_matrix = sparse.load_npz(os.path.join(processed_dir, "interaction_matrix.npz"))
            print("Loaded interaction matrix.")
        except:
            print("Warning: Could not load interaction matrix.")
            self.interaction_matrix = None
        
        # Load models
        self.load_models()
        
        # Load user sequences for sequence-aware recommendations
        try:
            self.user_sequences = {}
            train_features = pd.read_csv(os.path.join(processed_dir, "train_features.csv"), low_memory=True)
            sorted_df = train_features.sort_values(['user_id', 'timestamp'])
            
            for user_id, group in sorted_df.groupby('user_id'):
                self.user_sequences[user_id] = group['video_id'].tolist()
        except:
            print("Warning: Could not load user sequences for sequence-aware recommendations.")
            self.user_sequences = {}
    
    def load_models(self):
        """Load all trained models."""
        # Load collaborative filtering model
        try:
            self.als_model = ALSModel.load(os.path.join(self.models_dir, "als_model.pkl"))
            print("Loaded collaborative filtering model.")
        except Exception as e:
            print(f"Warning: Could not load collaborative filtering model: {e}")
            self.als_model = None
        
        # Load content-based model
        try:
            self.content_model = ContentBasedModel.load(os.path.join(self.models_dir, "content_model.pkl"))
            print("Loaded content-based model.")
        except Exception as e:
            print(f"Warning: Could not load content-based model: {e}")
            self.content_model = None
        
        # Load sequence-aware model
        try:
            self.seq_model = SequentialRules.load(os.path.join(self.models_dir, "sequential_model.pkl"))
            print("Loaded sequence-aware model.")
        except Exception as e:
            print(f"Warning: Could not load sequence-aware model: {e}")
            self.seq_model = None
        
        # Load hybrid model
        try:
            self.hybrid_model = LightGBMModel.load(os.path.join(self.models_dir, "hybrid_model"))
            print("Loaded hybrid model.")
        except Exception as e:
            print(f"Warning: Could not load hybrid model: {e}")
            self.hybrid_model = None
            
    def recommend_collaborative(self, user_id, n=10):
        """Generate collaborative filtering recommendations."""
        if self.als_model is None or self.interaction_matrix is None:
            return []
        
        if user_id not in self.user_indices:
            return []
        
        user_idx = self.user_indices[user_id]
        
        # Get user factors
        user_vector = self.als_model.user_factors[user_idx]
        
        # Compute scores for all items
        scores = np.dot(user_vector, self.als_model.item_factors.T) + self.als_model.global_mean
        
        # Exclude seen items
        seen_mask = self.interaction_matrix[user_idx].toarray().flatten() > 0
        scores[seen_mask] = -np.inf
        
        # Find top N items
        top_indices = np.argsort(scores)[-n:][::-1]
        
        # Convert to (item_id, score) format
        return [(self.index_to_item[idx], scores[idx]) for idx in top_indices]
    
    def recommend_content_based(self, user_id, n=10):
        """Generate content-based recommendations."""
        if self.content_model is None or user_id not in self.content_model.user_profiles:
            return []
        
        # Get list of seen items to exclude
        exclude_items = []
        if self.interaction_matrix is not None and user_id in self.user_indices:
            user_idx = self.user_indices[user_id]
            seen_items = self.interaction_matrix[user_idx].nonzero()[1]
            exclude_items = [self.index_to_item[idx] for idx in seen_items]
        
        # Use the content model's recommend method
        return self.content_model.recommend(user_id, n=n, exclude_items=exclude_items)
    
    def recommend_sequence(self, user_id, n=10):
        """Generate sequence-aware recommendations."""
        if self.seq_model is None or user_id not in self.user_sequences:
            return []
        
        # Get user's item sequence
        sequence = self.user_sequences[user_id]
        
        # If sequence is too short, return empty
        if len(sequence) < 2:
            return []
        
        # Get recommendations (scores are already normalized in the model)
        return self.seq_model.predict_next(sequence, k=n, normalize=True)
    
    def recommend(self, user_id, n=10, weights=None):
        """Generate recommendations using all available models."""
        # Default weights
        if weights is None:
            weights = {
                'collaborative': 0.4,
                'content': 0.3,
                'sequence': 0.2,
                'hybrid': 0.1
            }
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Get recommendations from each model
        cf_recs = self.recommend_collaborative(user_id, n=n*2)
        content_recs = self.recommend_content_based(user_id, n=n*2)
        seq_recs = self.recommend_sequence(user_id, n=n*2)
        
        # Combine recommendations
        all_items = {}
        
        # Add collaborative filtering recommendations
        for item_id, score in cf_recs:
            all_items[item_id] = all_items.get(item_id, 0) + score * weights['collaborative']
        
        # Add content-based recommendations
        for item_id, score in content_recs:
            all_items[item_id] = all_items.get(item_id, 0) + score * weights['content']
        
        # Add sequence-aware recommendations
        for item_id, score in seq_recs:
            # Scores are already normalized in the SequentialRules model
            all_items[item_id] = all_items.get(item_id, 0) + score * weights['sequence']
        
        # Sort by score and return top N
        sorted_items = sorted(all_items.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
    
    def generate_recommendations_for_all_users(self, users=None, n=10, weights=None):
        """Generate recommendations for all users or a subset of users."""
        if users is None:
            users = list(self.user_indices.keys())
        
        recommendations = {}
        
        for user_id in tqdm(users, desc="Generating recommendations"):
            recs = self.recommend(user_id, n=n, weights=weights)
            recommendations[user_id] = recs
        
        return recommendations
    
    def save_recommendations(self, recommendations, filepath):
        """Save recommendations to a file."""
        # Convert to DataFrame format
        rows = []
        
        for user_id, recs in recommendations.items():
            for rank, (item_id, score) in enumerate(recs):
                rows.append({
                    'user_id': user_id,
                    'video_id': item_id,
                    'rank': rank + 1,
                    'score': score
                })
        
        recommendations_df = pd.DataFrame(rows)
        
        # Save to CSV
        recommendations_df.to_csv(filepath, index=False)
        print(f"Saved recommendations to {filepath}")