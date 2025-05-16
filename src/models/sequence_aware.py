# Sequence-aware models for KuaiRec recommender system
from collections import defaultdict
import pickle
import numpy as np

class SequentialRules:
    """Simple sequence-aware model based on sequential rules."""
    
    def __init__(self, max_sequence_length=5, min_support=2):
        """Set up sequence model with length and support parameters."""
        self.max_sequence_length = max_sequence_length
        self.min_support = min_support
        self.item_sequences = defaultdict(int)
        self.item_to_items = defaultdict(lambda: defaultdict(int))
        self.max_count = 0  
        
    def fit(self, train_df, user_col='user_id', item_col='video_id', time_col='timestamp'):
        """Train model on user viewing sequences."""
        # Sort data by user and time
        sorted_df = train_df.sort_values([user_col, time_col])
        
        # Process each user's sequence
        for user_id, group in sorted_df.groupby(user_col):
            items = group[item_col].tolist()
            
            # Look at different sequence lengths
            for length in range(2, min(self.max_sequence_length + 1, len(items) + 1)):
                for i in range(len(items) - length + 1):
                    # Create a sequence
                    sequence = tuple(items[i:i+length])
                    
                    # Count this sequence
                    self.item_sequences[sequence] += 1
                    
                    # Count item-to-item transitions
                    for j in range(length - 1):
                        self.item_to_items[sequence[j]][sequence[j+1]] += 1
        
        # Keep only sequences with enough support
        self.item_sequences = {seq: count for seq, count in self.item_sequences.items() 
                              if count >= self.min_support}
        
        # Keep only transitions with enough support
        for item1 in list(self.item_to_items.keys()):
            self.item_to_items[item1] = {item2: count for item2, count in self.item_to_items[item1].items() 
                                       if count >= self.min_support}
            if not self.item_to_items[item1]:
                del self.item_to_items[item1]
        
        # Find max count for normalization
        self.max_count = self.min_support 
        if self.item_sequences:
            self.max_count = max(self.max_count, max(self.item_sequences.values()))
        
        for item, transitions in self.item_to_items.items():
            if transitions:
                self.max_count = max(self.max_count, max(transitions.values()))
                
        return self
    
    def predict_next(self, sequence, k=10, normalize=True):
        """
        Predict next items based on a viewing sequence.
        """
        if not sequence:
            return []
        
        # Focus on the most recent items
        recent_items = sequence[-self.max_sequence_length:]
        
        # Track scores for potential next items
        item_scores = defaultdict(float)
        
        # Check what usually follows the last item
        if recent_items[-1] in self.item_to_items:
            for item, count in self.item_to_items[recent_items[-1]].items():
                item_scores[item] += count
        
        # Check for longer patterns
        for length in range(2, min(self.max_sequence_length, len(recent_items)) + 1):
            seq = tuple(recent_items[-length:])
            if seq in self.item_sequences:
                # Find sequences that start with this pattern
                for full_seq, count in self.item_sequences.items():
                    if len(full_seq) > length and full_seq[:length] == seq:
                        item_scores[full_seq[length]] += count
        
        # Calculate similarity scores (cosine similarity-like normalization)
        if normalize and self.max_count > 0:
            for item in item_scores:
                item_scores[item] /= self.max_count
        
        # Return top items
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
    
    def save(self, filepath):
        """Save model to disk."""
        # Convert defaultdicts to regular dicts for saving
        item_to_items_dict = {}
        for item1, transitions in self.item_to_items.items():
            item_to_items_dict[item1] = dict(transitions)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'item_sequences': dict(self.item_sequences),
                'item_to_items': item_to_items_dict,
                'max_sequence_length': self.max_sequence_length,
                'min_support': self.min_support,
                'max_count': self.max_count
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new model with saved parameters
        model = cls(
            max_sequence_length=data['max_sequence_length'],
            min_support=data['min_support']
        )
        
        # Restore sequence counts
        model.item_sequences = defaultdict(int)
        model.item_sequences.update(data['item_sequences'])
        
        # Restore transition counts
        model.item_to_items = defaultdict(lambda: defaultdict(int))
        for item1, transitions in data['item_to_items'].items():
            for item2, count in transitions.items():
                model.item_to_items[item1][item2] = count
        
        # Restore max count if available
        if 'max_count' in data:
            model.max_count = data['max_count']
        else:
            # Calculate max count if not stored
            model.max_count = model.min_support
            if model.item_sequences:
                model.max_count = max(model.max_count, max(model.item_sequences.values()))
            
            for item, transitions in model.item_to_items.items():
                if transitions:
                    model.max_count = max(model.max_count, max(transitions.values()))
            
        return model 