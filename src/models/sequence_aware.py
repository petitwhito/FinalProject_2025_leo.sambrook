# Sequence-aware models for KuaiRec recommender system
from collections import defaultdict

class SequentialRules:
    """Simple sequence-aware model based on sequential rules."""
    
    def __init__(self, max_sequence_length=5, min_support=2):
        """
        Initialize sequential rules model.
        
        Parameters:
        -----------
        max_sequence_length : int
            Maximum sequence length to consider
        min_support : int
            Minimum support for a sequence to be considered
        """
        self.max_sequence_length = max_sequence_length
        self.min_support = min_support
        self.item_sequences = defaultdict(int)
        self.item_to_items = defaultdict(lambda: defaultdict(int))
        
    def fit(self, train_df, user_col='user_id', item_col='video_id', time_col='timestamp'):
        """
        Train the model on the training data.
        
        Parameters:
        -----------
        train_df : pandas.DataFrame
            Training data
        user_col : str
            Column name for user IDs
        item_col : str
            Column name for item IDs
        time_col : str
            Column name for timestamp
        """
        # Sort by user and timestamp
        sorted_df = train_df.sort_values([user_col, time_col])
        
        # Extract user sequences
        for user_id, group in sorted_df.groupby(user_col):
            items = group[item_col].tolist()
            
            # Consider sequences of different lengths
            for length in range(2, min(self.max_sequence_length + 1, len(items) + 1)):
                for i in range(len(items) - length + 1):
                    # Get the sequence and convert to tuple for hashability
                    sequence = tuple(items[i:i+length])
                    
                    # Update sequence count
                    self.item_sequences[sequence] += 1
                    
                    # Update transition counts
                    for j in range(length - 1):
                        self.item_to_items[sequence[j]][sequence[j+1]] += 1
        
        # Filter sequences by support
        self.item_sequences = {seq: count for seq, count in self.item_sequences.items() 
                              if count >= self.min_support}
        
        # Filter transitions by support
        for item1 in list(self.item_to_items.keys()):
            self.item_to_items[item1] = {item2: count for item2, count in self.item_to_items[item1].items() 
                                       if count >= self.min_support}
            if not self.item_to_items[item1]:
                del self.item_to_items[item1]
        
        return self
    
    def predict_next(self, sequence, k=10):
        """
        Predict the next items given a sequence.
        
        Parameters:
        -----------
        sequence : list
            Sequence of items
        k : int
            Number of recommendations to generate
            
        Returns:
        --------
        list
            List of (item_id, score) tuples
        """
        if not sequence:
            return []
        
        # Get the most recent items
        recent_items = sequence[-self.max_sequence_length:]
        
        # Calculate scores for each item
        item_scores = defaultdict(float)
        
        # Consider the most recent item
        if recent_items[-1] in self.item_to_items:
            for item, count in self.item_to_items[recent_items[-1]].items():
                item_scores[item] += count
        
        # Consider longer sequences if available
        for length in range(2, min(self.max_sequence_length, len(recent_items)) + 1):
            seq = tuple(recent_items[-length:])
            if seq in self.item_sequences:
                # Look for sequences that start with this sequence
                for full_seq, count in self.item_sequences.items():
                    if len(full_seq) > length and full_seq[:length] == seq:
                        item_scores[full_seq[length]] += count
        
        # Sort by score and return top k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
    
    def save(self, filepath):
        """Save the model to a file."""
        import pickle
        
        # Convert defaultdict with lambda to regular dict for pickling
        item_to_items_dict = {}
        for item1, transitions in self.item_to_items.items():
            item_to_items_dict[item1] = dict(transitions)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'item_sequences': dict(self.item_sequences),
                'item_to_items': item_to_items_dict,
                'max_sequence_length': self.max_sequence_length,
                'min_support': self.min_support
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        import pickle
        from collections import defaultdict
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            max_sequence_length=data['max_sequence_length'],
            min_support=data['min_support']
        )
        
        # Convert back to defaultdict
        model.item_sequences = defaultdict(int)
        model.item_sequences.update(data['item_sequences'])
        
        model.item_to_items = defaultdict(lambda: defaultdict(int))
        for item1, transitions in data['item_to_items'].items():
            for item2, count in transitions.items():
                model.item_to_items[item1][item2] = count
            
        return model 