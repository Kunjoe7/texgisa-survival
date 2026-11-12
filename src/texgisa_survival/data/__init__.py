# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Data processing utilities for survival analysis.
"""

from typing import Optional, Tuple, Union, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .manager import DataManager


class SurvivalDataset:
    """
    Container for survival analysis data with preprocessing utilities.

    Examples
    --------
    >>> from texgisa_survival.data import SurvivalDataset
    >>> dataset = SurvivalDataset(data, time_col='time', event_col='event')
    >>> dataset.normalize()
    >>> dataset.handle_missing(strategy='median')
    >>> X_train, X_test, y_train, y_test = dataset.split(test_size=0.2)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str,
        event_col: str,
        feature_cols: Optional[List[str]] = None,
    ):
        """
        Initialize survival dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        time_col : str
            Name of the time column
        event_col : str
            Name of the event column
        feature_cols : list of str, optional
            Feature columns to use. If None, uses all except time and event.
        """
        self.data = data.copy()
        self.time_col = time_col
        self.event_col = event_col

        if feature_cols is None:
            self.feature_cols = [col for col in data.columns if col not in [time_col, event_col]]
        else:
            self.feature_cols = feature_cols

        self.scaler = None
        self.label_encoders = {}
        self.is_normalized = False

    @property
    def X(self) -> pd.DataFrame:
        """Get feature matrix."""
        return self.data[self.feature_cols]

    @property
    def time(self) -> pd.Series:
        """Get survival times."""
        return self.data[self.time_col]

    @property
    def event(self) -> pd.Series:
        """Get event indicators."""
        return self.data[self.event_col]

    def normalize(
        self, method: str = "standard", exclude_cols: Optional[List[str]] = None
    ) -> "SurvivalDataset":
        """
        Normalize numerical features.

        Parameters
        ----------
        method : str
            Normalization method ('standard', 'minmax')
        exclude_cols : list of str, optional
            Columns to exclude from normalization

        Returns
        -------
        self : SurvivalDataset
        """
        if exclude_cols is None:
            exclude_cols = []

        numeric_cols = [
            col
            for col in self.feature_cols
            if self.data[col].dtype in ["float64", "int64"] and col not in exclude_cols
        ]

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        self.is_normalized = True

        return self

    def encode_categorical(
        self, categorical_cols: Optional[List[str]] = None, method: str = "label"
    ) -> "SurvivalDataset":
        """
        Encode categorical variables.

        Parameters
        ----------
        categorical_cols : list of str, optional
            Categorical columns to encode. If None, auto-detects.
        method : str
            Encoding method ('label', 'onehot')

        Returns
        -------
        self : SurvivalDataset
        """
        if categorical_cols is None:
            categorical_cols = [
                col for col in self.feature_cols if self.data[col].dtype == "object"
            ]

        if method == "label":
            for col in categorical_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
        elif method == "onehot":
            self.data = pd.get_dummies(self.data, columns=categorical_cols)
            # Update feature columns
            self.feature_cols = [
                col for col in self.data.columns if col not in [self.time_col, self.event_col]
            ]
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        return self

    def handle_missing(
        self, strategy: str = "median", fill_value: Optional[float] = None
    ) -> "SurvivalDataset":
        """
        Handle missing values.

        Parameters
        ----------
        strategy : str
            Strategy for handling missing values ('median', 'mean', 'drop', 'constant')
        fill_value : float, optional
            Value to use when strategy='constant'

        Returns
        -------
        self : SurvivalDataset
        """
        if strategy == "drop":
            self.data = self.data.dropna(subset=self.feature_cols + [self.time_col, self.event_col])
        elif strategy in ["median", "mean"]:
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = [
                col for col in self.feature_cols if self.data[col].dtype in ["float64", "int64"]
            ]
            self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        elif strategy == "constant":
            self.data[self.feature_cols] = self.data[self.feature_cols].fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self

    def filter_by_time(
        self, min_time: Optional[float] = None, max_time: Optional[float] = None
    ) -> "SurvivalDataset":
        """
        Filter data by survival time.

        Parameters
        ----------
        min_time : float, optional
            Minimum survival time
        max_time : float, optional
            Maximum survival time

        Returns
        -------
        self : SurvivalDataset
        """
        mask = pd.Series([True] * len(self.data))

        if min_time is not None:
            mask &= self.data[self.time_col] >= min_time
        if max_time is not None:
            mask &= self.data[self.time_col] <= max_time

        self.data = self.data[mask].reset_index(drop=True)

        return self

    def split(
        self, test_size: float = 0.2, stratify_by_event: bool = True, random_state: int = 42
    ) -> Tuple:
        """
        Split data into train and test sets.

        Parameters
        ----------
        test_size : float
            Proportion of data for testing
        stratify_by_event : bool
            Whether to stratify by event indicator
        random_state : int
            Random seed

        Returns
        -------
        X_train, X_test, time_train, time_test, event_train, event_test : tuple
        """
        stratify = self.event if stratify_by_event else None

        return train_test_split(
            self.X,
            self.time,
            self.event,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    def get_summary(self) -> dict:
        """
        Get dataset summary statistics.

        Returns
        -------
        summary : dict
            Summary statistics
        """
        return {
            "n_samples": len(self.data),
            "n_features": len(self.feature_cols),
            "event_rate": self.event.mean(),
            "median_time": self.time.median(),
            "mean_time": self.time.mean(),
            "std_time": self.time.std(),
            "min_time": self.time.min(),
            "max_time": self.time.max(),
            "missing_values": self.data[self.feature_cols].isnull().sum().sum(),
            "feature_types": {
                "numeric": sum(
                    self.data[col].dtype in ["float64", "int64"] for col in self.feature_cols
                ),
                "categorical": sum(self.data[col].dtype == "object" for col in self.feature_cols),
            },
        }


def create_dataset(
    data: Union[pd.DataFrame, str],
    time_col: str,
    event_col: str,
    feature_cols: Optional[List[str]] = None,
    normalize: bool = True,
    handle_missing: str = "median",
    encode_categorical: bool = True,
) -> SurvivalDataset:
    """
    Create a preprocessed survival dataset.

    Parameters
    ----------
    data : pd.DataFrame or str
        Input data or path to CSV file
    time_col : str
        Name of the time column
    event_col : str
        Name of the event column
    feature_cols : list of str, optional
        Feature columns to use
    normalize : bool
        Whether to normalize features
    handle_missing : str
        Strategy for missing values
    encode_categorical : bool
        Whether to encode categorical variables

    Returns
    -------
    dataset : SurvivalDataset
        Preprocessed dataset
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    dataset = SurvivalDataset(data, time_col, event_col, feature_cols)

    if handle_missing:
        dataset.handle_missing(strategy=handle_missing)

    if encode_categorical:
        dataset.encode_categorical()

    if normalize:
        dataset.normalize()

    return dataset
