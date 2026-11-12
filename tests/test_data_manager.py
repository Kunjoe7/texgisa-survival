# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""Tests for DataManager."""

import pandas as pd
import pytest

from texgisa_survival.data.manager import DataManager


@pytest.fixture
def sample_df():
    return pd.DataFrame({"duration": [10, 20, 30], "event": [1, 0, 1], "age": [50, 60, 70]})


class TestDataManager:
    def test_initial_state(self):
        dm = DataManager()
        assert dm.get_data() is None
        assert dm.get_current_file_name() is None

    def test_load_data(self, sample_df):
        dm = DataManager()
        dm.load_data(sample_df, file_name="test.csv")
        assert dm.get_data() is not None
        assert dm.get_current_file_name() == "test.csv"
        assert dm.current_file_name == "test.csv"
        assert len(dm.get_data()) == 3

    def test_get_current_dataframe_alias(self, sample_df):
        dm = DataManager()
        dm.load_data(sample_df)
        assert dm.get_current_dataframe() is dm.get_data()

    def test_load_multimodal(self, sample_df):
        dm = DataManager()
        image_df = pd.DataFrame({"img_feat": [1.0, 2.0, 3.0]})
        dm.load_multimodal_data(tabular_df=sample_df, image_df=image_df)
        assert dm.tabular_df is sample_df
        assert dm.image_df is image_df
        assert dm.sensor_df is None
        # First non-None is used as _data
        assert dm.get_data() is sample_df

    def test_load_multimodal_image_only(self):
        dm = DataManager()
        image_df = pd.DataFrame({"img_feat": [1.0]})
        dm.load_multimodal_data(image_df=image_df)
        assert dm.get_data() is image_df

    def test_get_data_summary_empty(self):
        dm = DataManager()
        summary = dm.get_data_summary()
        assert "error" in summary

    def test_get_data_summary_populated(self, sample_df):
        dm = DataManager()
        dm.load_data(sample_df, file_name="test.csv")
        summary = dm.get_data_summary()
        assert summary["file_name"] == "test.csv"
        assert summary["num_rows"] == 3
        assert summary["num_columns"] == 3
        assert "column_names" in summary
        assert "missing_values" in summary

    def test_get_data_summary_multimodal(self, sample_df):
        dm = DataManager()
        dm.load_data(sample_df, file_name="main.csv")
        dm.load_multimodal_data(tabular_df=sample_df, sensor_df=pd.DataFrame({"s": [1]}))
        summary = dm.get_data_summary()
        assert "modalities" in summary
        assert "tabular" in summary["modalities"]
        assert "sensor" in summary["modalities"]
