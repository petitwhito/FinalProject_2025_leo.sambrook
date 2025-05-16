# Hybrid recommendation models for KuaiRec recommender system

import lightgbm as lgb

class LightGBMModel:
    """Hybrid model using LightGBM."""
    
    def __init__(self, params=None):
        """
        Initialize LightGBM model.
        
        Parameters:
        -----------
        params : dict, optional
            LightGBM parameters
        """
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
        """
        Train the model on the training data.
        
        Parameters:
        -----------
        train_features_df : pandas.DataFrame
            DataFrame with training features
        feature_columns : list, optional
            List of columns to use as features
        num_boost_round : int
            Number of boosting rounds
        """
        # If no feature columns specified, use all except ID columns and target
        if feature_columns is None:
            exclude_cols = ['user_id', 'video_id', 'watch_ratio', 'time', 'date', 'timestamp', 'datetime']
            feature_columns = [col for col in train_features_df.columns if col not in exclude_cols]
        
        self.feature_names = feature_columns
        
        # Extract features and target
        X = train_features_df[feature_columns]
        y = train_features_df['watch_ratio']
        
        # Convert categorical variables to numeric
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Split into train and validation
        valid_size = int(0.2 * len(X))
        X_train, X_valid = X.iloc[:-valid_size], X.iloc[-valid_size:]
        y_train, y_valid = y.iloc[:-valid_size], y.iloc[-valid_size:]
        
        # Create datasets
        categorical_features = X.select_dtypes(include=['category']).columns.tolist()
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=categorical_features)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data]
        )
        
        return self
    
    def predict(self, features):
        """
        Predict watch ratio.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features for prediction
            
        Returns:
        --------
        numpy.ndarray
            Predicted watch ratios
        """
        # Prepare features
        X = features[self.feature_names].copy()
        
        # Convert categorical variables to numeric
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Make predictions
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save the model to a file."""
        # Save model
        self.model.save_model(filepath + '.model')
        
        # Save feature names
        import pickle
        with open(filepath + '.features', 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        model = cls()
        
        # Load model
        model.model = lgb.Booster(model_file=filepath + '.model')
        
        # Load feature names
        import pickle
        with open(filepath + '.features', 'rb') as f:
            model.feature_names = pickle.load(f)
        
        return model 