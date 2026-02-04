"""
CiRA ME - Feature Extraction Service
Handles TSFresh and Custom DSP feature extraction
Provides 40+ features for comprehensive signal analysis
"""

import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.fft import fft, fftfreq

# Import data loader to access session data
from .data_loader import _data_sessions

# Global storage for extracted features
_feature_sessions: Dict[str, Dict] = {}


def _autocorr(x, lag):
    """Compute autocorrelation at a specific lag."""
    n = len(x)
    if lag >= n:
        return 0.0
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.0
    return np.sum((x[:n-lag] - mean) * (x[lag:] - mean)) / (n * var)


def _binned_entropy(x, num_bins=10):
    """Compute binned entropy of a signal."""
    hist, _ = np.histogram(x, bins=num_bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-10))


class FeatureExtractor:
    """Service for extracting features from windowed time-series data."""

    # TSFresh-like statistical features (25 features)
    TSFRESH_FEATURES = {
        'mean': lambda x: np.mean(x, axis=0),
        'std': lambda x: np.std(x, axis=0),
        'min': lambda x: np.min(x, axis=0),
        'max': lambda x: np.max(x, axis=0),
        'median': lambda x: np.median(x, axis=0),
        'sum': lambda x: np.sum(x, axis=0),
        'variance': lambda x: np.var(x, axis=0),
        'skewness': lambda x: stats.skew(x, axis=0),
        'kurtosis': lambda x: stats.kurtosis(x, axis=0),
        'abs_energy': lambda x: np.sum(x ** 2, axis=0),
        'root_mean_square': lambda x: np.sqrt(np.mean(x ** 2, axis=0)),
        'mean_abs_change': lambda x: np.mean(np.abs(np.diff(x, axis=0)), axis=0),
        'mean_change': lambda x: np.mean(np.diff(x, axis=0), axis=0),
        'count_above_mean': lambda x: np.sum(x > np.mean(x, axis=0), axis=0),
        'count_below_mean': lambda x: np.sum(x < np.mean(x, axis=0), axis=0),
        'first_location_of_maximum': lambda x: np.argmax(x, axis=0) / len(x),
        'first_location_of_minimum': lambda x: np.argmin(x, axis=0) / len(x),
        'last_location_of_maximum': lambda x: (len(x) - 1 - np.argmax(x[::-1], axis=0)) / len(x),
        'last_location_of_minimum': lambda x: (len(x) - 1 - np.argmin(x[::-1], axis=0)) / len(x),
        'percentage_of_reoccurring_values': lambda x: np.array([
            len(np.unique(col)) / len(col) for col in x.T
        ]),
        'sum_of_reoccurring_values': lambda x: np.array([
            np.sum([v for v in col if np.sum(col == v) > 1]) for col in x.T
        ]),
        # Additional TSFresh features
        'abs_sum_of_changes': lambda x: np.sum(np.abs(np.diff(x, axis=0)), axis=0),
        'range': lambda x: np.max(x, axis=0) - np.min(x, axis=0),
        'interquartile_range': lambda x: np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0),
        'mean_second_derivative': lambda x: np.mean(np.diff(np.diff(x, axis=0), axis=0), axis=0) if len(x) > 2 else np.zeros(x.shape[1]),
    }

    def __init__(self):
        pass

    def _compute_dsp_features(self, window: np.ndarray, sampling_rate: float = 100.0) -> Dict[str, np.ndarray]:
        """Compute custom DSP features for a window (19 features)."""
        features = {}

        # Time-domain features
        features['rms'] = np.sqrt(np.mean(window ** 2, axis=0))
        features['peak_to_peak'] = np.max(window, axis=0) - np.min(window, axis=0)

        rms = features['rms']
        rms_safe = np.where(rms == 0, 1e-10, rms)

        features['crest_factor'] = np.max(np.abs(window), axis=0) / rms_safe

        mean_abs = np.mean(np.abs(window), axis=0)
        mean_abs_safe = np.where(mean_abs == 0, 1e-10, mean_abs)

        features['shape_factor'] = rms_safe / mean_abs_safe
        features['impulse_factor'] = np.max(np.abs(window), axis=0) / mean_abs_safe

        mean_sqrt = np.mean(np.sqrt(np.abs(window)), axis=0) ** 2
        mean_sqrt_safe = np.where(mean_sqrt == 0, 1e-10, mean_sqrt)

        features['margin_factor'] = np.max(np.abs(window), axis=0) / mean_sqrt_safe

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
        features['zero_crossing_rate'] = zero_crossings / (len(window) - 1)

        # Additional time-domain features
        features['autocorr_lag1'] = np.array([_autocorr(col, 1) for col in window.T])
        features['autocorr_lag5'] = np.array([_autocorr(col, 5) for col in window.T])
        features['binned_entropy'] = np.array([_binned_entropy(col) for col in window.T])

        # Frequency-domain features (FFT)
        n_samples = len(window)
        freqs = fftfreq(n_samples, 1.0 / sampling_rate)
        positive_freq_mask = freqs >= 0

        spectral_features = []
        for col_idx in range(window.shape[1]):
            col_data = window[:, col_idx]

            # FFT
            fft_vals = fft(col_data)
            fft_magnitude = np.abs(fft_vals)[positive_freq_mask]
            fft_freqs = freqs[positive_freq_mask]

            # Normalize magnitude
            total_power = np.sum(fft_magnitude ** 2)
            total_power_safe = total_power if total_power > 0 else 1e-10
            normalized_power = fft_magnitude ** 2 / total_power_safe

            # Spectral centroid
            spectral_centroid = np.sum(fft_freqs * normalized_power)

            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((fft_freqs - spectral_centroid) ** 2) * normalized_power))

            # Spectral rolloff (95%)
            cumsum = np.cumsum(normalized_power)
            rolloff_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
            spectral_rolloff = fft_freqs[min(rolloff_idx, len(fft_freqs) - 1)]

            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
            arithmetic_mean = np.mean(fft_magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

            # Spectral entropy
            normalized_power_safe = normalized_power + 1e-10
            spectral_entropy = -np.sum(normalized_power_safe * np.log2(normalized_power_safe))

            # Peak frequency
            peak_freq_idx = np.argmax(fft_magnitude)
            peak_frequency = fft_freqs[peak_freq_idx]

            # Spectral skewness and kurtosis
            spectral_skewness = stats.skew(fft_magnitude)
            spectral_kurtosis = stats.kurtosis(fft_magnitude)

            # Band power (low, mid, high)
            low_mask = (fft_freqs >= 0) & (fft_freqs < sampling_rate / 6)
            mid_mask = (fft_freqs >= sampling_rate / 6) & (fft_freqs < sampling_rate / 3)
            high_mask = (fft_freqs >= sampling_rate / 3) & (fft_freqs <= sampling_rate / 2)

            band_power_low = np.sum(fft_magnitude[low_mask] ** 2) / total_power_safe
            band_power_mid = np.sum(fft_magnitude[mid_mask] ** 2) / total_power_safe
            band_power_high = np.sum(fft_magnitude[high_mask] ** 2) / total_power_safe

            spectral_features.append({
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_rolloff': spectral_rolloff,
                'spectral_flatness': spectral_flatness,
                'spectral_entropy': spectral_entropy,
                'peak_frequency': peak_frequency,
                'spectral_skewness': spectral_skewness,
                'spectral_kurtosis': spectral_kurtosis,
                'band_power_low': band_power_low,
                'band_power_mid': band_power_mid,
                'band_power_high': band_power_high,
            })

        # Aggregate spectral features
        for key in spectral_features[0].keys():
            features[key] = np.array([sf[key] for sf in spectral_features])

        return features

    def extract(
        self,
        session_id: str,
        selected_features: Optional[List[str]] = None,
        include_tsfresh: bool = True,
        include_dsp: bool = True,
        sampling_rate: float = 100.0
    ) -> Dict[str, Any]:
        """
        Extract features from windowed data.

        Args:
            session_id: Session ID containing windowed data
            selected_features: List of specific features to extract (None = all)
            include_tsfresh: Include TSFresh statistical features
            include_dsp: Include custom DSP features
            sampling_rate: Sampling rate for frequency analysis

        Returns:
            Feature extraction results with session ID
        """
        session = _data_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if 'windows' not in session:
            raise ValueError("Session does not contain windowed data. Apply windowing first.")

        windows = session['windows']
        labels = session.get('labels')
        categories = session.get('categories')
        metadata = session['metadata']

        num_windows = len(windows)
        num_channels = windows[0].shape[1]
        sensor_columns = metadata.get('sensor_columns', [f'ch_{i}' for i in range(num_channels)])

        all_features = []
        feature_names = []

        for window_idx, window in enumerate(windows):
            window_features = {}

            # TSFresh features
            if include_tsfresh:
                for feat_name, feat_func in self.TSFRESH_FEATURES.items():
                    if selected_features is None or feat_name in selected_features:
                        try:
                            values = feat_func(window)
                            for ch_idx, val in enumerate(values):
                                col_name = f"{feat_name}_{sensor_columns[ch_idx]}"
                                window_features[col_name] = val
                                if window_idx == 0:
                                    feature_names.append(col_name)
                        except Exception:
                            pass

            # DSP features
            if include_dsp:
                dsp_features = self._compute_dsp_features(window, sampling_rate)
                for feat_name, values in dsp_features.items():
                    if selected_features is None or feat_name in selected_features:
                        for ch_idx, val in enumerate(values):
                            col_name = f"{feat_name}_{sensor_columns[ch_idx]}"
                            window_features[col_name] = val
                            if window_idx == 0:
                                feature_names.append(col_name)

            all_features.append(window_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)

        # Store in session
        feature_session_id = f"features_{session_id}"
        _feature_sessions[feature_session_id] = {
            'features': features_df,
            'labels': labels,
            'categories': categories,
            'feature_names': feature_names,
            'metadata': {
                **metadata,
                'num_features': len(feature_names),
                'feature_categories': {
                    'tsfresh': include_tsfresh,
                    'dsp': include_dsp
                }
            }
        }

        return {
            'session_id': feature_session_id,
            'num_windows': num_windows,
            'num_features': len(feature_names),
            'feature_names': feature_names,
            'preview': features_df.head(5).to_dict(orient='records')
        }

    def recommend_features(self, session_id: str, mode: str = 'anomaly') -> Dict[str, Any]:
        """
        Recommend features based on data characteristics and mode.

        This provides rule-based recommendations. LLM integration can be added
        for more sophisticated analysis.
        """
        session = _data_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        recommendations = []
        reasoning = []

        # Always recommended
        base_features = ['mean', 'std', 'rms', 'min', 'max']
        recommendations.extend(base_features)
        reasoning.append("Basic statistical features provide fundamental signal characteristics.")

        if mode == 'anomaly':
            # Anomaly detection benefits from distribution and energy features
            anomaly_features = [
                'kurtosis', 'skewness', 'peak_to_peak', 'crest_factor',
                'spectral_entropy', 'zero_crossing_rate', 'abs_energy'
            ]
            recommendations.extend(anomaly_features)
            reasoning.append("Distribution features (kurtosis, skewness) help detect unusual patterns.")
            reasoning.append("Energy features identify abnormal signal power levels.")

        elif mode == 'classification':
            # Classification benefits from frequency and shape features
            classification_features = [
                'spectral_centroid', 'spectral_bandwidth', 'peak_frequency',
                'band_power_low', 'band_power_mid', 'band_power_high',
                'shape_factor', 'impulse_factor'
            ]
            recommendations.extend(classification_features)
            reasoning.append("Frequency features help distinguish different signal classes.")
            reasoning.append("Shape factors capture waveform characteristics unique to each class.")

        return {
            'recommended_features': list(set(recommendations)),
            'reasoning': reasoning,
            'mode': mode,
            'total_recommended': len(set(recommendations))
        }

    def get_importance(self, feature_session_id: str, training_session_id: str) -> Dict[str, Any]:
        """Get feature importance scores from a trained model."""
        # This would integrate with the training service to get importance scores
        # For now, return placeholder
        feature_session = _feature_sessions.get(feature_session_id)
        if not feature_session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        feature_names = feature_session['feature_names']

        # Placeholder importance scores
        importance_scores = {name: np.random.random() for name in feature_names}

        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'feature_importance': dict(sorted_features[:20]),  # Top 20
            'total_features': len(feature_names)
        }

    @staticmethod
    def get_features_for_training(feature_session_id: str) -> tuple:
        """Get feature matrix, labels, and categories for ML training.

        Returns:
            Tuple of (X, y, categories) where categories may be None.
        """
        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        X = session['features'].values
        y = session['labels']
        categories = session.get('categories')

        return X, y, categories

    def get_feature_preview(self, feature_session_id: str, num_rows: int = 100) -> Dict[str, Any]:
        """
        Get feature preview data for visualization.

        Args:
            feature_session_id: Feature session ID
            num_rows: Number of rows to return for preview

        Returns:
            Feature preview with statistics
        """
        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features']
        labels = session['labels']
        feature_names = session['feature_names']

        # Compute feature statistics
        feature_stats = {}
        for col in features_df.columns:
            values = features_df[col].values
            feature_stats[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }

        # Get preview rows
        preview_df = features_df.head(num_rows).copy()
        if labels is not None:
            preview_df['label'] = labels[:num_rows]

        # Replace NaN/inf with 0 for JSON serialization
        preview_df = preview_df.replace([np.inf, -np.inf], 0).fillna(0)

        # Compute label counts
        label_counts = {}
        if labels is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_counts = {str(label): int(count) for label, count in zip(unique_labels, counts)}

        return {
            'session_id': feature_session_id,
            'num_features': len(feature_names),
            'num_windows': len(features_df),
            'columns': list(preview_df.columns),
            'feature_names': feature_names,
            'feature_stats': feature_stats,
            'preview': preview_df.to_dict(orient='records'),
            'labels': list(np.unique(labels)) if labels is not None else None,
            'label_counts': label_counts
        }

    def get_feature_distribution(self, feature_session_id: str, feature_name: str, bins: int = 20) -> Dict[str, Any]:
        """
        Get distribution data for a specific feature (for histogram visualization).

        Args:
            feature_session_id: Feature session ID
            feature_name: Name of the feature to analyze
            bins: Number of histogram bins

        Returns:
            Distribution data for the feature
        """
        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features']
        labels = session['labels']

        if feature_name not in features_df.columns:
            raise ValueError(f"Feature not found: {feature_name}")

        values = features_df[feature_name].values

        # Compute histogram
        hist, bin_edges = np.histogram(values, bins=bins)

        # Per-label distribution if labels exist
        label_distributions = {}
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_mask = labels == label
                label_values = values[label_mask]
                label_hist, _ = np.histogram(label_values, bins=bin_edges)
                label_distributions[str(label)] = label_hist.tolist()

        return {
            'feature_name': feature_name,
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'label_distributions': label_distributions,
            'statistics': {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }
        }
