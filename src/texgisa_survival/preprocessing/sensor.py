# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Sensor data preprocessing for survival analysis.

This module processes time-series sensor data and extracts features
suitable for survival analysis models.
"""

import os
import io
import zipfile
import tempfile
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
from pathlib import Path
import warnings


class SensorProcessor:
    """
    Process sensor time-series data for survival analysis.
    
    This processor handles:
    - Multiple sensor channels
    - Variable length sequences
    - Feature extraction (statistical, frequency domain)
    - Resampling to target frequency
    
    Parameters
    ----------
    target_hz : float, optional
        Target sampling frequency for resampling
    max_rows : int, optional
        Maximum rows to read per file (0 = no limit)
    feature_types : list of str
        Types of features to extract: 'statistical', 'frequency', 'both'
    verbose : bool
        Print processing information
        
    Examples
    --------
    >>> processor = SensorProcessor(target_hz=100.0)
    >>> 
    >>> # Process from ZIP file
    >>> features = processor.process_zip('sensors.zip', 'labels.csv')
    >>> 
    >>> # Process from directory
    >>> features = processor.process_directory('sensor_data/', 'labels.csv')
    >>> 
    >>> # Process single file
    >>> features = processor.process_file('sensor1.csv')
    """
    
    NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", 
                      "float16", "float32", "float64")
    
    def __init__(self, 
                 target_hz: Optional[float] = None,
                 max_rows: int = 0,
                 feature_types: str = 'both',
                 verbose: bool = True):
        self.target_hz = target_hz
        self.max_rows = max_rows
        self.feature_types = feature_types
        self.verbose = verbose
        
    def process_zip(self, 
                   zip_path: Union[str, Path, io.BytesIO],
                   labels_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Process sensor data from ZIP file.
        
        Parameters
        ----------
        zip_path : str, Path, or BytesIO
            Path to ZIP file containing sensor data files
        labels_path : str, Path, or DataFrame
            Path to labels CSV or DataFrame with columns: file, duration, event
            
        Returns
        -------
        features_df : pd.DataFrame
            DataFrame with extracted features and survival labels
        """
        # Extract to temporary directory
        with tempfile.TemporaryDirectory(prefix="sensors_") as tmp_dir:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            
            return self.process_directory(tmp_dir, labels_path)
    
    def process_directory(self,
                         directory: Union[str, Path],
                         labels: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Process all sensor files in a directory.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing sensor data files
        labels : str, Path, or DataFrame
            Labels data with columns: file, duration, event
            
        Returns
        -------
        features_df : pd.DataFrame
            DataFrame with extracted features and survival labels
        """
        directory = Path(directory)
        
        # Load labels
        if isinstance(labels, pd.DataFrame):
            labels_df = labels
        else:
            labels_df = pd.read_csv(labels)
        
        # Validate labels
        required_cols = {'file', 'duration', 'event'}
        if not required_cols.issubset(labels_df.columns):
            missing = required_cols - set(labels_df.columns)
            raise ValueError(f"Labels missing required columns: {missing}")
        
        # Find sensor files
        sensor_files = self._scan_sensor_files(directory)
        
        if self.verbose:
            print(f"Found {len(sensor_files)} sensor files")
        
        # Process each file
        all_features = []
        
        for rel_path, abs_path in sensor_files:
            # Get base name for matching with labels
            basename = os.path.splitext(os.path.basename(rel_path))[0]
            
            # Find corresponding label
            label_row = labels_df[labels_df['file'].str.contains(basename, na=False)]
            
            if label_row.empty:
                if self.verbose:
                    print(f"Warning: No label found for {basename}, skipping")
                continue
            
            # Extract features
            try:
                features = self.process_file(abs_path)
                
                # Add labels
                features['duration'] = label_row['duration'].values[0]
                features['event'] = label_row['event'].values[0]
                features['file'] = basename
                
                # Add any extra columns from labels
                extra_cols = [col for col in label_row.columns 
                             if col not in ['file', 'duration', 'event']]
                for col in extra_cols:
                    features[col] = label_row[col].values[0]
                
                all_features.append(features)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {basename}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No sensor files could be processed successfully")
        
        # Combine all features
        features_df = pd.DataFrame(all_features)
        
        # Reorder columns: labels first, then features
        label_cols = ['duration', 'event', 'file'] + extra_cols
        feature_cols = [col for col in features_df.columns if col not in label_cols]
        features_df = features_df[label_cols + feature_cols]
        
        if self.verbose:
            print(f"Processed {len(features_df)} samples with {len(feature_cols)} features each")
        
        return features_df
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    time_col: Optional[str] = None) -> Dict[str, float]:
        """
        Process a single sensor file and extract features.
        
        Parameters
        ----------
        file_path : str or Path
            Path to sensor data file (CSV, Parquet, or TXT)
        time_col : str, optional
            Name of timestamp column for resampling
            
        Returns
        -------
        features : dict
            Dictionary of extracted features
        """
        # Read sensor data
        df = self._read_sensor_file(file_path)
        
        # Resample if needed
        if self.target_hz and time_col and time_col in df.columns:
            df = self._resample_to_target_hz(df, time_col, self.target_hz)
        
        # Get numeric columns (sensor channels)
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in self.NUMERIC_DTYPES]
        
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in {file_path}")
        
        # Extract features for each channel
        features = {}
        
        for col in numeric_cols:
            channel_data = df[col].values
            
            # Statistical features
            if self.feature_types in ['statistical', 'both']:
                stats = self._extract_statistical_features(channel_data, col)
                features.update(stats)
            
            # Frequency domain features
            if self.feature_types in ['frequency', 'both']:
                freq = self._extract_frequency_features(channel_data, col)
                features.update(freq)
        
        return features
    
    def _scan_sensor_files(self, 
                          directory: Path,
                          extensions: set = {'.csv', '.parquet', '.txt'}) -> List[Tuple[str, Path]]:
        """Scan directory for sensor files."""
        files = []
        
        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(directory)
                    files.append((str(rel_path), file_path))
        
        return sorted(files)
    
    def _read_sensor_file(self, file_path: Path) -> pd.DataFrame:
        """Read sensor data from various formats."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif ext in ['.csv', '.txt']:
            # Try different separators
            for sep in [',', '\t', ' ', ';']:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            else:
                df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Limit rows if specified
        if self.max_rows > 0 and len(df) > self.max_rows:
            df = df.iloc[:self.max_rows]
        
        return df
    
    def _resample_to_target_hz(self, 
                               df: pd.DataFrame,
                               time_col: str,
                               target_hz: float) -> pd.DataFrame:
        """Resample sensor data to target frequency."""
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Set time as index
        df = df.set_index(time_col)
        
        # Calculate target period
        target_period = f'{int(1000/target_hz)}ms'
        
        # Resample
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in self.NUMERIC_DTYPES]
        
        resampled = df[numeric_cols].resample(target_period).mean()
        
        return resampled.reset_index()
    
    def _extract_statistical_features(self, 
                                     data: np.ndarray,
                                     prefix: str) -> Dict[str, float]:
        """Extract statistical features from sensor channel."""
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            return {f'sens_{prefix}_missing': 1.0}
        
        features = {
            f'sens_{prefix}_mean': np.mean(data),
            f'sens_{prefix}_std': np.std(data),
            f'sens_{prefix}_min': np.min(data),
            f'sens_{prefix}_max': np.max(data),
            f'sens_{prefix}_median': np.median(data),
            f'sens_{prefix}_q25': np.percentile(data, 25),
            f'sens_{prefix}_q75': np.percentile(data, 75),
            f'sens_{prefix}_iqr': np.percentile(data, 75) - np.percentile(data, 25),
            f'sens_{prefix}_skew': self._safe_skew(data),
            f'sens_{prefix}_kurtosis': self._safe_kurtosis(data),
            f'sens_{prefix}_rms': np.sqrt(np.mean(data**2)),
            f'sens_{prefix}_zero_cross': self._zero_crossing_rate(data),
            f'sens_{prefix}_peak_count': self._count_peaks(data),
        }
        
        # Time-domain features
        if len(data) > 1:
            diff = np.diff(data)
            features[f'sens_{prefix}_mean_abs_diff'] = np.mean(np.abs(diff))
            features[f'sens_{prefix}_std_diff'] = np.std(diff)
        
        return features
    
    def _extract_frequency_features(self,
                                   data: np.ndarray,
                                   prefix: str,
                                   n_freq_bins: int = 10) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return {}
        
        # Apply FFT
        fft_vals = np.fft.rfft(data)
        fft_power = np.abs(fft_vals) ** 2
        
        # Normalize
        if fft_power.sum() > 0:
            fft_power = fft_power / fft_power.sum()
        
        # Bin the spectrum
        n_bins = min(n_freq_bins, len(fft_power))
        bin_size = len(fft_power) // n_bins
        
        features = {}
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(fft_power)
            bin_power = fft_power[start_idx:end_idx].sum()
            features[f'sens_{prefix}_fft_bin{i:02d}'] = bin_power
        
        # Spectral features
        freqs = np.fft.rfftfreq(len(data))
        
        if len(fft_power) > 0 and fft_power.sum() > 0:
            # Spectral centroid
            spectral_centroid = np.sum(freqs * fft_power) / np.sum(fft_power)
            features[f'sens_{prefix}_spectral_centroid'] = spectral_centroid
            
            # Spectral spread
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_power) / np.sum(fft_power))
            features[f'sens_{prefix}_spectral_spread'] = spectral_spread
            
            # Dominant frequency
            dominant_freq = freqs[np.argmax(fft_power)]
            features[f'sens_{prefix}_dominant_freq'] = dominant_freq
        
        return features
    
    def _safe_skew(self, data: np.ndarray) -> float:
        """Calculate skewness safely."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        if len(data) < 2:
            return 0.0
        
        signs = np.sign(data)
        diff = np.diff(signs)
        crossings = np.where(diff != 0)[0]
        
        return len(crossings) / len(data)
    
    def _count_peaks(self, data: np.ndarray, prominence: float = 0.1) -> int:
        """Count peaks in the signal."""
        if len(data) < 3:
            return 0
        
        # Simple peak detection
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if data[i] - min(data[i-1], data[i+1]) > prominence * np.std(data):
                    peaks.append(i)
        
        return len(peaks)