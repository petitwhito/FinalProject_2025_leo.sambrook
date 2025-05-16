import lightgbm as lgb
import pickle

class LightGBMModel:
    """Hybrid model using LightGBM."""
    
    def __init__(self, params=None):
        """Set up LightGBM model with default or custom parameters."""
        # Default parameters
        if params is None:
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'device_type': 'gpu',
                'verbose': 100
            }
        else:
            self.params = params
        
        self.model = None
        self.feature_names = None
    
    def fit(self, train_features_df, feature_columns=None,
           num_boost_round=1000):
        """Train model on the given features."""
        # Figure out which columns to use as features
        if feature_columns is None:
            exclude_cols = ['user_id', 'video_id', 'watch_ratio', 'time', 'date', 'timestamp', 'datetime']
            feature_columns = [col for col in train_features_df.columns if col not in exclude_cols]
        
        self.feature_names = feature_columns
        
        # Get features and target
        X = train_features_df[feature_columns]
        y = train_features_df['watch_ratio']
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Create train/validation split
        valid_size = int(0.2 * len(X))
        X_train, X_valid = X.iloc[:-valid_size], X.iloc[-valid_size:]
        y_train, y_valid = y.iloc[:-valid_size], y.iloc[-valid_size:]
        
        # Set up LightGBM datasets
        categorical_features = X.select_dtypes(include=['category']).columns.tolist()
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=categorical_features)
        
        # Train the model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data]
        )
        
        return self
    
    def predict(self, features):
        """Make predictions for the given features."""
        # Prepare feature data
        X = features[self.feature_names].copy()
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Generate predictions
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to disk."""
        # Save the model itself
        self.model.save_model(filepath + '.model')
        
        # Save feature names separately
        with open(filepath + '.features', 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        model = cls()
        
        # Load the model
        model.model = lgb.Booster(model_file=filepath + '.model')
        
        # Load feature names
        with open(filepath + '.features', 'rb') as f:
            model.feature_names = pickle.load(f)
        
        return model 