import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import pickle

class ALSModel:
    """SVD-based collaborative filtering model."""
    
    def __init__(self, factors=100):
        """Set up model with specified number of latent factors."""
        self.factors = factors
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, interaction_matrix):
        """Train model using SVD on the interaction matrix."""
        # Handle input matrix format
        if isinstance(interaction_matrix, sparse.spmatrix):
            matrix = interaction_matrix.copy()
        else:
            matrix = sparse.csr_matrix(interaction_matrix)
        
        # Save for later use
        self.interaction_matrix = matrix
        
        # Get global mean of non-zero values
        nonzero_mask = matrix.nonzero()
        self.global_mean = matrix[nonzero_mask].mean()
        
        # Center the data
        matrix_centered = matrix.copy()
        nonzero_mask = matrix_centered.nonzero()
        matrix_centered[nonzero_mask] = matrix_centered[nonzero_mask] - self.global_mean
        
        # Run SVD
        u, s, vt = svds(matrix_centered, k=self.factors)
        
        # Build factors with the decomposition results
        s_diag = np.diag(s)
        self.user_factors = np.dot(u, np.sqrt(s_diag))
        self.item_factors = np.dot(np.sqrt(s_diag), vt).T
        
        return self
    
    def predict(self, user_idx, item_idx):
        """Predict rating for a user-item pair."""
        if self.user_factors is None:
            return self.global_mean
        
        # Simple dot product of latent factors + global mean
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx].T)
        prediction += self.global_mean
        
        return prediction
    
    def recommend(self, user_idx, n=10, exclude_seen=True):
        """Get top-N recommendations for a user."""
        if self.user_factors is None:
            return []
        
        # Get this user's latent factors
        user_vector = self.user_factors[user_idx]
        
        # Score all items
        scores = np.dot(user_vector, self.item_factors.T) + self.global_mean
        
        # Don't recommend items they've already seen
        if exclude_seen:
            seen_mask = self.interaction_matrix[user_idx].toarray().flatten() > 0
            scores[seen_mask] = -np.inf
        
        # Find top items
        top_indices = np.argsort(scores)[-n:][::-1]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'global_mean': self.global_mean,
                'factors': self.factors
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(factors=data['factors'])
        model.user_factors = data['user_factors']
        model.item_factors = data['item_factors']
        model.global_mean = data['global_mean']
        return model