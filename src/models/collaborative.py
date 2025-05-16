# Collaborative filtering models for KuaiRec recommender system
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

class ALSModel:
    """SVD-based model for collaborative filtering using scipy."""
    
    def __init__(self, factors=100):
        """
        Initialize SVD model.
        
        Parameters:
        -----------
        factors : int
            Number of latent factors
        """
        self.factors = factors
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        
    def fit(self, interaction_matrix):
        """
        Train the model on the interaction matrix using SVD.
        
        Parameters:
        -----------
        interaction_matrix : scipy.sparse.csr_matrix or numpy.ndarray
            User-item interaction matrix
        """
        # Fill NaN values with zeros
        if isinstance(interaction_matrix, sparse.spmatrix):
            # Already a sparse matrix
            matrix = interaction_matrix.copy()
        else:
            # Convert to sparse matrix
            matrix = sparse.csr_matrix(interaction_matrix)
        
        # Save the interaction matrix for later use
        self.interaction_matrix = matrix
        
        # Calculate global mean
        nonzero_mask = matrix.nonzero()
        self.global_mean = matrix[nonzero_mask].mean()
        
        # Center the data by subtracting the mean
        matrix_centered = matrix.copy()
        nonzero_mask = matrix_centered.nonzero()
        matrix_centered[nonzero_mask] = matrix_centered[nonzero_mask] - self.global_mean
        
        # Compute SVD
        u, s, vt = svds(matrix_centered, k=self.factors)
        
        # Convert s to a diagonal matrix
        s_diag = np.diag(s)
        
        # Store the latent factors
        self.user_factors = np.dot(u, np.sqrt(s_diag))
        self.item_factors = np.dot(np.sqrt(s_diag), vt).T
        
        return self
    
    def predict(self, user_idx, item_idx):
        """
        Predict the rating for a given user and item.
        
        Parameters:
        -----------
        user_idx : int
            User index
        item_idx : int
            Item index
            
        Returns:
        --------
        float
            Predicted rating
        """
        if self.user_factors is None:
            return self.global_mean
        
        # Calculate prediction
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx].T)
        prediction += self.global_mean
        
        return prediction
    
    def recommend(self, user_idx, n=10, exclude_seen=True):
        """
        Generate recommendations for a user.
        
        Parameters:
        -----------
        user_idx : int
            User index
        n : int
            Number of recommendations
        exclude_seen : bool
            Whether to exclude already seen items
            
        Returns:
        --------
        list
            List of (item_idx, score) tuples
        """
        if self.user_factors is None:
            return []
        
        # Get user factors
        user_vector = self.user_factors[user_idx]
        
        # Compute scores for all items
        scores = np.dot(user_vector, self.item_factors.T) + self.global_mean
        
        # Get top N items
        if exclude_seen:
            # Create a mask for items the user has already interacted with
            seen_mask = self.interaction_matrix[user_idx].toarray().flatten() > 0
            scores[seen_mask] = -np.inf
        
        top_indices = np.argsort(scores)[-n:][::-1]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def save(self, filepath):
        """Save the model to a file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'global_mean': self.global_mean,
                'factors': self.factors
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(factors=data['factors'])
        model.user_factors = data['user_factors']
        model.item_factors = data['item_factors']
        model.global_mean = data['global_mean']
        return model