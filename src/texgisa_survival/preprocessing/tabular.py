# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Tabular data preprocessing utilities for survival analysis.

This module provides standard preprocessing for tabular survival data.
"""

from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings


class TabularProcessor:
    """
    Standard preprocessing for tabular survival data.
    
    This processor handles:
    - Missing value imputation
    - Feature scaling/normalization
    - Categorical encoding
    - Feature selection
    - Outlier handling
    
    Parameters
    ----------
    scaler_type : str
        Type of scaler: 'standard', 'minmax', 'robust', or None
    imputation_strategy : str
        Strategy for missing values: 'mean', 'median', 'most_frequent', 'constant'
    handle_outliers : bool
        Whether to handle outliers
    outlier_threshold : float
        Z-score threshold for outlier detection
    verbose : bool
        Print processing information
        
    Examples
    --------
    >>> processor = TabularProcessor(scaler_type='standard')
    >>> 
    >>> # Fit and transform training data
    >>> X_train_processed = processor.fit_transform(X_train)
    >>> 
    >>> # Transform test data
    >>> X_test_processed = processor.transform(X_test)
    >>> 
    >>> # Process with survival labels
    >>> data_processed = processor.process_survival_data(data, 'time', 'event')
    """
    
    def __init__(self,
                 scaler_type: str = 'standard',
                 imputation_strategy: str = 'median',
                 handle_outliers: bool = False,
                 outlier_threshold: float = 3.0,
                 verbose: bool = True):
        
        self.scaler_type = scaler_type
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose
        
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
        self._is_fitted = False
    
    def fit_transform(self, 
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Optional[np.ndarray] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit preprocessor and transform data.
        
        Parameters
        ----------
        X : DataFrame or array
            Input features
        y : array, optional
            Target variable (not used, for sklearn compatibility)
            
        Returns
        -------
        X_transformed : DataFrame or array
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Optional[np.ndarray] = None) -> 'TabularProcessor':
        """
        Fit the preprocessor.
        
        Parameters
        ----------
        X : DataFrame or array
            Input features
        y : array, optional
            Target variable (not used)
            
        Returns
        -------
        self : TabularProcessor
            Fitted processor
        """
        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            X_numeric = X[self.numeric_features].values if self.numeric_features else None
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.numeric_features = self.feature_names
            self.categorical_features = []
            X_numeric = X
        
        # Fit imputer
        if X_numeric is not None and self.imputation_strategy:
            self.imputer = SimpleImputer(strategy=self.imputation_strategy)
            self.imputer.fit(X_numeric)
        
        # Fit scaler
        if X_numeric is not None and self.scaler_type:
            # Apply imputation first if needed
            if self.imputer:
                X_numeric = self.imputer.transform(X_numeric)
            
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
            
            self.scaler.fit(X_numeric)
        
        self._is_fitted = True
        
        if self.verbose:
            print(f"Fitted preprocessor on {len(self.feature_names)} features")
            if self.numeric_features:
                print(f"  Numeric features: {len(self.numeric_features)}")
            if self.categorical_features:
                print(f"  Categorical features: {len(self.categorical_features)}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        X : DataFrame or array
            Input features
            
        Returns
        -------
        X_transformed : DataFrame or array
            Transformed features
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return_df = isinstance(X, pd.DataFrame)
        
        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_features].values if self.numeric_features else None
            X_cat = X[self.categorical_features] if self.categorical_features else None
        else:
            X_numeric = X
            X_cat = None
        
        # Transform numeric features
        if X_numeric is not None:
            # Impute
            if self.imputer:
                X_numeric = self.imputer.transform(X_numeric)
            
            # Handle outliers
            if self.handle_outliers and self.scaler_type == 'standard':
                X_numeric = self._handle_outliers(X_numeric)
            
            # Scale
            if self.scaler:
                X_numeric = self.scaler.transform(X_numeric)
        
        # Combine features
        if return_df:
            result = pd.DataFrame()
            
            if X_numeric is not None and self.numeric_features:
                numeric_df = pd.DataFrame(X_numeric, 
                                         columns=self.numeric_features,
                                         index=X.index if isinstance(X, pd.DataFrame) else None)
                result = pd.concat([result, numeric_df], axis=1)
            
            if X_cat is not None and self.categorical_features:
                result = pd.concat([result, X_cat.reset_index(drop=True)], axis=1)
            
            # Restore original column order
            if isinstance(X, pd.DataFrame):
                cols = [col for col in X.columns if col in result.columns]
                result = result[cols]
            
            return result
        else:
            return X_numeric
    
    def process_survival_data(self,
                             data: pd.DataFrame,
                             time_col: str,
                             event_col: str,
                             feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process complete survival dataset.
        
        Parameters
        ----------
        data : DataFrame
            Complete dataset with features and survival information
        time_col : str
            Name of time/duration column
        event_col : str
            Name of event indicator column
        feature_cols : list, optional
            Feature columns to process (default: all except time and event)
            
        Returns
        -------
        processed_data : DataFrame
            Processed dataset with original structure
        """
        data = data.copy()
        
        # Identify feature columns
        if feature_cols is None:
            feature_cols = [col for col in data.columns 
                          if col not in [time_col, event_col]]
        
        # Separate features and labels
        X = data[feature_cols]
        survival_info = data[[time_col, event_col]]
        
        # Process features
        X_processed = self.fit_transform(X)
        
        # Combine back
        if isinstance(X_processed, np.ndarray):
            X_processed = pd.DataFrame(X_processed, 
                                      columns=feature_cols,
                                      index=data.index)
        
        processed_data = pd.concat([survival_info, X_processed], axis=1)
        
        # Add any remaining columns
        other_cols = [col for col in data.columns 
                     if col not in processed_data.columns]
        if other_cols:
            processed_data = pd.concat([processed_data, data[other_cols]], axis=1)
        
        if self.verbose:
            print(f"Processed survival data: {len(processed_data)} samples, {len(feature_cols)} features")
            print(f"Event rate: {data[event_col].mean():.1%}")
            print(f"Median time: {data[time_col].median():.2f}")
        
        return processed_data
    
    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Handle outliers using z-score clipping."""
        if self.scaler and hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            # Calculate z-scores
            z_scores = np.abs((X - self.scaler.mean_) / (self.scaler.scale_ + 1e-10))
            
            # Clip outliers
            X_clipped = X.copy()
            outlier_mask = z_scores > self.outlier_threshold
            
            if outlier_mask.any():
                # Clip to threshold
                X_clipped[outlier_mask] = np.sign(X[outlier_mask] - self.scaler.mean_[None, :]) * \
                                         self.outlier_threshold * self.scaler.scale_[None, :] + \
                                         self.scaler.mean_[None, :]
                
                if self.verbose:
                    n_outliers = outlier_mask.sum()
                    print(f"Clipped {n_outliers} outlier values")
            
            return X_clipped
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        return self.feature_names or []
    
    def get_params(self) -> Dict[str, Any]:
        """Get preprocessing parameters."""
        return {
            'scaler_type': self.scaler_type,
            'imputation_strategy': self.imputation_strategy,
            'handle_outliers': self.handle_outliers,
            'outlier_threshold': self.outlier_threshold,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_numeric': len(self.numeric_features) if self.numeric_features else 0,
            'n_categorical': len(self.categorical_features) if self.categorical_features else 0,
        }