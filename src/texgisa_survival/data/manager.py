# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
DataManager — lightweight dataset container for survival analysis workflows.

Ported from the SAWEB application with all Streamlit-specific code removed.
"""

from typing import Optional

import pandas as pd


class DataManager:
    """Manage the active dataset for a survival analysis session.

    Supports single-modality (tabular) data as well as optional multimodal
    data (tabular, image, sensor).

    Examples
    --------
    >>> from texgisa_survival.data import DataManager
    >>> dm = DataManager()
    >>> dm.load_data(pd.read_csv("data.csv"), file_name="data.csv")
    >>> dm.get_data().head()
    """

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._file_name: Optional[str] = None
        self.current_file_name: Optional[str] = None

        self.tabular_df: Optional[pd.DataFrame] = None
        self.image_df: Optional[pd.DataFrame] = None
        self.sensor_df: Optional[pd.DataFrame] = None

    def load_data(self, data: pd.DataFrame, file_name: Optional[str] = None) -> None:
        """Load a pandas DataFrame into the manager.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to manage.
        file_name : str, optional
            Descriptive name for the dataset.
        """
        name = file_name or "uploaded_dataset"
        self._data = data
        self._file_name = name
        self.current_file_name = name

    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the loaded DataFrame (or *None* if nothing has been loaded)."""
        return self._data

    def get_current_dataframe(self) -> Optional[pd.DataFrame]:
        """Alias for :meth:`get_data`."""
        return self._data

    def get_current_file_name(self) -> Optional[str]:
        """Return the name associated with the loaded dataset."""
        return self._file_name

    def load_multimodal_data(
        self,
        *,
        tabular_df: Optional[pd.DataFrame] = None,
        image_df: Optional[pd.DataFrame] = None,
        sensor_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Load optional DataFrames for different data modalities.

        Parameters
        ----------
        tabular_df, image_df, sensor_df : pd.DataFrame, optional
            DataFrames for each modality.  The first non-``None`` DataFrame is
            also exposed via :meth:`get_data` for backward compatibility.
        """
        self.tabular_df = tabular_df
        self.image_df = image_df
        self.sensor_df = sensor_df

        for candidate in (tabular_df, image_df, sensor_df):
            if candidate is not None:
                self._data = candidate
                break

    def get_data_summary(self) -> dict:
        """Return a summary of the loaded data.

        Returns
        -------
        dict
            Keys include ``file_name``, ``num_rows``, ``num_columns``,
            ``column_names``, ``missing_values``, and optionally ``modalities``.
        """
        if self._data is None and not any(
            df is not None for df in (self.tabular_df, self.image_df, self.sensor_df)
        ):
            return {"error": "No data has been loaded."}

        summary: dict = {"file_name": self._file_name}
        if self._data is not None:
            summary.update(
                {
                    "num_rows": int(self._data.shape[0]),
                    "num_columns": int(self._data.shape[1]),
                    "column_names": self._data.columns.tolist(),
                    "missing_values": self._data.isnull().sum().to_dict(),
                }
            )

        modalities = {}
        for name, df in [
            ("tabular", self.tabular_df),
            ("image", self.image_df),
            ("sensor", self.sensor_df),
        ]:
            if df is not None:
                modalities[name] = {
                    "num_rows": int(df.shape[0]),
                    "num_columns": int(df.shape[1]),
                    "missing_values": df.isnull().sum().to_dict(),
                }

        if modalities:
            summary["modalities"] = modalities

        return summary
