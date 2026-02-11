"""
CiRA ME - Feature Extraction Service
Handles TSFresh and Custom DSP feature extraction
Supports both:
- Real tsfresh library (800+ features with FRESH feature selection)
- Lightweight custom implementation (44 features for edge deployment)
Includes intelligent feature selection with statistical and hypothesis testing methods
"""

import numpy as np
import pandas as pd
import uuid
import warnings
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Real tsfresh imports
try:
    from tsfresh import extract_features as tsfresh_extract_features
    from tsfresh import select_features as tsfresh_select_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    warnings.warn("tsfresh not installed. Only lightweight features available. Install with: pip install tsfresh")

# Import data loader to access session data
from .data_loader import _data_sessions

# Global storage for extracted features
_feature_sessions: Dict[str, Dict] = {}

# Global storage for feature selection results
_selection_sessions: Dict[str, Dict] = {}


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

    def extract_tsfresh(
        self,
        session_id: str,
        feature_set: str = 'efficient',  # 'minimal', 'efficient', 'comprehensive'
        n_jobs: int = 1,
        chunksize: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract features using the REAL tsfresh library.

        This provides 800+ features with the comprehensive set, including:
        - Statistical features (mean, std, skewness, kurtosis, etc.)
        - Autocorrelation at multiple lags
        - Partial autocorrelation
        - FFT coefficients (multiple)
        - AR model coefficients
        - Wavelet coefficients (CWT)
        - Change quantiles
        - Energy ratio by chunks
        - Linear trend features
        - Number of peaks/crossings
        - And many more...

        Args:
            session_id: Session ID containing windowed data
            feature_set: Feature extraction configuration
                - 'minimal': ~10 features per column (fast)
                - 'efficient': ~100 features per column (balanced)
                - 'comprehensive': ~800 features per column (thorough)
            n_jobs: Number of parallel jobs (-1 for all cores)
            chunksize: Chunk size for parallel processing

        Returns:
            Feature extraction results with session ID
        """
        if not TSFRESH_AVAILABLE:
            raise ImportError(
                "tsfresh library not installed. Install with: pip install tsfresh\n"
                "Or use the lightweight 'extract' method instead."
            )

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
        window_size = windows[0].shape[0]
        sensor_columns = metadata.get('sensor_columns', [f'ch_{i}' for i in range(num_channels)])

        print(f"[tsfresh] Extracting features from {num_windows} windows, {num_channels} channels")
        print(f"[tsfresh] Feature set: {feature_set}")

        # Select feature calculation settings
        if feature_set == 'minimal':
            fc_parameters = MinimalFCParameters()
            expected_features = "~10 per column"
        elif feature_set == 'efficient':
            fc_parameters = EfficientFCParameters()
            expected_features = "~100 per column"
        else:  # comprehensive
            fc_parameters = ComprehensiveFCParameters()
            expected_features = "~800 per column"

        print(f"[tsfresh] Expected features: {expected_features}")

        # Convert windows to tsfresh format (long format DataFrame)
        # tsfresh expects: id, time, value columns
        records = []
        for window_idx, window in enumerate(windows):
            for time_idx in range(window_size):
                for ch_idx in range(num_channels):
                    records.append({
                        'id': window_idx,
                        'time': time_idx,
                        'kind': sensor_columns[ch_idx],
                        'value': float(window[time_idx, ch_idx])
                    })

        df_long = pd.DataFrame(records)

        print(f"[tsfresh] Data prepared: {len(df_long)} rows")

        # Extract features using tsfresh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_df = tsfresh_extract_features(
                df_long,
                column_id='id',
                column_sort='time',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=fc_parameters,
                n_jobs=n_jobs,
                chunksize=chunksize,
                disable_progressbar=True
            )

        # Handle NaN/Inf values
        features_df = impute(features_df)

        # Get feature names
        feature_names = list(features_df.columns)

        print(f"[tsfresh] Extracted {len(feature_names)} features")

        # Store in session
        feature_session_id = f"tsfresh_{session_id}"
        _feature_sessions[feature_session_id] = {
            'features': features_df,
            'labels': labels,
            'categories': categories,
            'feature_names': feature_names,
            'metadata': {
                **metadata,
                'num_features': len(feature_names),
                'extraction_method': 'tsfresh',
                'feature_set': feature_set,
                'feature_categories': {
                    'tsfresh_real': True,
                    'feature_set': feature_set
                }
            }
        }

        return {
            'session_id': feature_session_id,
            'num_windows': num_windows,
            'num_features': len(feature_names),
            'feature_names': feature_names,
            'feature_set': feature_set,
            'extraction_method': 'tsfresh',
            'preview': features_df.head(5).to_dict(orient='records')
        }

    def select_features_fresh(
        self,
        feature_session_id: str,
        fdr_level: float = 0.05,
        multiclass: bool = True,
        n_significant: int = 1,
        ml_task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Select features using tsfresh's FRESH algorithm.

        FRESH = FeatuRe Extraction based on Scalable Hypothesis tests

        This method:
        1. Performs appropriate statistical tests for each feature based on its type
        2. Applies Benjamini-Hochberg procedure for multiple testing correction
        3. Returns features with p-value below the FDR threshold

        Statistical tests used (automatically selected):
        - Kolmogorov-Smirnov test (for continuous features)
        - Mann-Whitney U test
        - Fisher's exact test (for binary features)
        - Welch's t-test
        - Chi-squared test

        Args:
            feature_session_id: Session ID with extracted features
            fdr_level: False Discovery Rate threshold (default 0.05)
            multiclass: Whether to handle multiclass targets
            n_significant: Minimum number of significant features to keep
            ml_task: 'classification' or 'regression'

        Returns:
            Selection results with selected features, p-values, and relevance table
        """
        if not TSFRESH_AVAILABLE:
            raise ImportError(
                "tsfresh library not installed. Install with: pip install tsfresh\n"
                "Use 'select_features' method for sklearn-based selection instead."
            )

        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features'].copy()
        labels = session['labels']

        if labels is None:
            raise ValueError("Labels required for FRESH feature selection")

        original_features = list(features_df.columns)
        y = pd.Series(labels)

        print(f"[FRESH] Starting feature selection on {len(original_features)} features")
        print(f"[FRESH] FDR level: {fdr_level}, ML task: {ml_task}")
        print(f"[FRESH] Labels: {len(y)} samples, {len(y.unique())} unique classes")

        # Apply tsfresh's FRESH feature selection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Select features using hypothesis testing with FDR correction
            features_filtered = tsfresh_select_features(
                features_df,
                y,
                ml_task=ml_task,
                multiclass=multiclass,
                n_significant=n_significant,
                fdr_level=fdr_level
            )

        selected_features = list(features_filtered.columns)

        print(f"[FRESH] Selected {len(selected_features)} features (from {len(original_features)})")

        # Compute feature relevance scores (using correlation as proxy since tsfresh
        # doesn't expose p-values directly in the simple API)
        relevance_scores = {}
        for feat in selected_features:
            if feat in features_df.columns:
                # Use absolute correlation with encoded labels as relevance proxy
                y_encoded = pd.factorize(y)[0]
                corr = abs(features_df[feat].corr(pd.Series(y_encoded)))
                relevance_scores[feat] = float(corr) if not np.isnan(corr) else 0.0

        # Normalize scores
        max_score = max(relevance_scores.values()) if relevance_scores else 1.0
        if max_score > 0:
            relevance_scores = {k: v / max_score for k, v in relevance_scores.items()}

        # Sort by relevance
        sorted_features = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

        # Store selection session
        selection_session_id = f"fresh_{feature_session_id}"
        _selection_sessions[selection_session_id] = {
            'feature_session_id': feature_session_id,
            'selected_features': selected_features,
            'relevance_scores': relevance_scores,
            'method': 'fresh',
            'fdr_level': fdr_level,
            'original_count': len(original_features),
            'final_count': len(selected_features)
        }

        return {
            'session_id': selection_session_id,
            'selected_features': selected_features,
            'relevance_scores': {f: relevance_scores.get(f, 0) for f in selected_features},
            'all_scores': dict(sorted_features),
            'selection_log': [
                f"Applied FRESH algorithm with FDR level {fdr_level}",
                f"Performed hypothesis testing with Benjamini-Hochberg correction",
                f"Selected {len(selected_features)} statistically significant features",
                f"Reduction: {len(original_features)} → {len(selected_features)} ({100 * len(selected_features) / len(original_features):.1f}%)"
            ],
            'original_count': len(original_features),
            'final_count': len(selected_features),
            'method': 'fresh',
            'fdr_level': fdr_level
        }

    def select_features_fresh_combined(
        self,
        feature_session_id: str,
        fdr_level: float = 0.05,
        n_features: int = 20,
        ml_task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Chained feature selection: FRESH + target count.

        Step 1: Apply FRESH to get statistically significant features
        Step 2: If more than n_features remain, reduce using mutual information ranking

        Args:
            feature_session_id: Session ID with extracted features
            fdr_level: False Discovery Rate threshold for FRESH
            n_features: Target number of features after second selection
            ml_task: 'classification' or 'regression'

        Returns:
            Selection results with selected features and selection log
        """
        if not TSFRESH_AVAILABLE:
            raise ImportError(
                "tsfresh library not installed. Install with: pip install tsfresh\n"
                "Use 'select_features' method for sklearn-based selection instead."
            )

        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features'].copy()
        labels = session['labels']

        if labels is None:
            raise ValueError("Labels required for feature selection")

        original_features = list(features_df.columns)
        y = pd.Series(labels)

        print(f"[FRESH+Combined] Starting chained selection on {len(original_features)} features")
        print(f"[FRESH+Combined] Step 1: FRESH with FDR level {fdr_level}")
        print(f"[FRESH+Combined] Step 2: Reduce to {n_features} features")

        selection_log = []

        # Step 1: Apply FRESH
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_after_fresh = tsfresh_select_features(
                features_df,
                y,
                ml_task=ml_task,
                multiclass=True,
                n_significant=1,
                fdr_level=fdr_level
            )

        fresh_features = list(features_after_fresh.columns)
        selection_log.append(f"Step 1 (FRESH): {len(original_features)} → {len(fresh_features)} features (FDR={fdr_level})")
        print(f"[FRESH+Combined] After FRESH: {len(fresh_features)} features")

        # Step 2: If we have more features than target, reduce using mutual information
        if len(fresh_features) <= n_features:
            # Already at or below target, no further reduction needed
            final_features = fresh_features
            selection_log.append(f"Step 2: No reduction needed (already at {len(fresh_features)} ≤ {n_features})")
        else:
            # Use mutual information to rank and select top n_features
            X_fresh = features_after_fresh.values
            y_encoded = pd.factorize(y)[0]

            # Compute mutual information scores
            mi_scores = mutual_info_classif(X_fresh, y_encoded, random_state=42)
            mi_ranking = sorted(zip(fresh_features, mi_scores), key=lambda x: x[1], reverse=True)

            # Select top n_features
            final_features = [f for f, _ in mi_ranking[:n_features]]
            selection_log.append(f"Step 2 (MI ranking): {len(fresh_features)} → {n_features} features")
            print(f"[FRESH+Combined] After MI ranking: {len(final_features)} features")

        # Compute relevance scores for final features
        relevance_scores = {}
        y_encoded = pd.factorize(y)[0]
        for feat in final_features:
            if feat in features_df.columns:
                corr = abs(features_df[feat].corr(pd.Series(y_encoded)))
                relevance_scores[feat] = float(corr) if not np.isnan(corr) else 0.0

        # Normalize scores
        max_score = max(relevance_scores.values()) if relevance_scores else 1.0
        if max_score > 0:
            relevance_scores = {k: v / max_score for k, v in relevance_scores.items()}

        # Sort by relevance
        sorted_features = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

        selection_log.append(f"Final: {len(final_features)} features selected")

        # Store selection session
        selection_session_id = f"fresh_combined_{feature_session_id}"
        _selection_sessions[selection_session_id] = {
            'feature_session_id': feature_session_id,
            'selected_features': final_features,
            'relevance_scores': relevance_scores,
            'method': 'fresh_combined',
            'fdr_level': fdr_level,
            'target_features': n_features,
            'original_count': len(original_features),
            'after_fresh_count': len(fresh_features),
            'final_count': len(final_features)
        }

        return {
            'session_id': selection_session_id,
            'selected_features': final_features,
            'importance_scores': relevance_scores,
            'relevance_scores': {f: relevance_scores.get(f, 0) for f in final_features},
            'all_scores': dict(sorted_features),
            'selection_log': selection_log,
            'original_count': len(original_features),
            'after_fresh_count': len(fresh_features),
            'final_count': len(final_features),
            'method': 'fresh_combined',
            'fdr_level': fdr_level,
            'target_features': n_features
        }

    @staticmethod
    def get_available_feature_sets() -> Dict[str, Any]:
        """Get information about available feature extraction options."""
        return {
            'tsfresh_available': TSFRESH_AVAILABLE,
            'extraction_methods': {
                'lightweight': {
                    'name': 'Lightweight (Custom)',
                    'description': '44 curated features optimized for edge deployment',
                    'features_per_channel': 44,
                    'speed': 'fast',
                    'includes': ['statistical', 'dsp', 'spectral']
                },
                'tsfresh_minimal': {
                    'name': 'tsfresh Minimal',
                    'description': '~10 basic statistical features per column',
                    'features_per_channel': 10,
                    'speed': 'fast',
                    'requires': 'tsfresh'
                },
                'tsfresh_efficient': {
                    'name': 'tsfresh Efficient',
                    'description': '~100 features per column (balanced)',
                    'features_per_channel': 100,
                    'speed': 'medium',
                    'requires': 'tsfresh'
                },
                'tsfresh_comprehensive': {
                    'name': 'tsfresh Comprehensive',
                    'description': '~800 features per column including FFT, wavelets, AR models',
                    'features_per_channel': 800,
                    'speed': 'slow',
                    'requires': 'tsfresh'
                }
            },
            'selection_methods': {
                'combined': {
                    'name': 'Combined (sklearn)',
                    'description': 'Variance + correlation + mutual info + ANOVA',
                    'hypothesis_testing': False
                },
                'fresh': {
                    'name': 'FRESH (tsfresh)',
                    'description': 'Hypothesis testing with Benjamini-Hochberg FDR correction',
                    'hypothesis_testing': True,
                    'requires': 'tsfresh'
                },
                'fresh_combined': {
                    'name': 'FRESH + Target Count',
                    'description': 'FRESH filtering then reduce to target count using MI ranking',
                    'hypothesis_testing': True,
                    'requires': 'tsfresh'
                }
            }
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

    def select_features(
        self,
        feature_session_id: str,
        method: str = 'combined',
        n_features: int = 15,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Intelligent feature selection from extracted features.

        Methods:
        - 'variance': Remove low-variance features
        - 'correlation': Remove highly correlated features
        - 'mutual_info': Select by mutual information with labels
        - 'anova': Select by ANOVA F-score
        - 'combined': Apply all methods in sequence (recommended)

        Args:
            feature_session_id: Session ID with extracted features
            method: Selection method
            n_features: Target number of features to select
            variance_threshold: Minimum variance threshold
            correlation_threshold: Maximum correlation to allow

        Returns:
            Selection results with selected features and reasoning
        """
        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features'].copy()
        labels = session['labels']
        original_features = list(features_df.columns)

        selection_log = []
        removed_features = {}

        # Step 1: Remove constant/near-constant features (variance filter)
        if method in ['variance', 'combined']:
            # Normalize first for fair variance comparison
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df)
            variances = np.var(scaled_features, axis=0)

            low_variance_mask = variances < variance_threshold
            low_var_features = [f for f, m in zip(features_df.columns, low_variance_mask) if m]

            if low_var_features:
                features_df = features_df.drop(columns=low_var_features)
                removed_features['low_variance'] = low_var_features
                selection_log.append(f"Removed {len(low_var_features)} low-variance features")

        # Step 2: Remove highly correlated features
        if method in ['correlation', 'combined'] and len(features_df.columns) > 1:
            corr_matrix = features_df.corr().abs()

            # Find highly correlated pairs
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            high_corr_features = []
            for col in upper_tri.columns:
                if any(upper_tri[col] > correlation_threshold):
                    # Keep the feature with higher variance
                    correlated_with = upper_tri.index[upper_tri[col] > correlation_threshold].tolist()
                    for corr_feature in correlated_with:
                        if corr_feature not in high_corr_features:
                            # Compare variances
                            if features_df[col].var() >= features_df[corr_feature].var():
                                high_corr_features.append(corr_feature)
                            else:
                                high_corr_features.append(col)
                                break

            high_corr_features = list(set(high_corr_features))
            if high_corr_features:
                features_df = features_df.drop(columns=high_corr_features)
                removed_features['high_correlation'] = high_corr_features
                selection_log.append(f"Removed {len(high_corr_features)} highly correlated features")

        # Step 3: Score remaining features by importance
        feature_scores = {}

        if labels is not None and len(np.unique(labels)) > 1:
            # Use mutual information for scoring
            if method in ['mutual_info', 'combined']:
                try:
                    mi_scores = mutual_info_classif(
                        features_df.values,
                        labels,
                        random_state=42,
                        n_neighbors=5
                    )
                    for feat, score in zip(features_df.columns, mi_scores):
                        feature_scores[feat] = feature_scores.get(feat, 0) + score
                    selection_log.append("Computed mutual information scores")
                except Exception as e:
                    selection_log.append(f"MI scoring failed: {str(e)}")

            # Use ANOVA F-score
            if method in ['anova', 'combined']:
                try:
                    f_scores, _ = f_classif(features_df.values, labels)
                    # Normalize scores
                    f_scores = np.nan_to_num(f_scores, nan=0.0)
                    if f_scores.max() > 0:
                        f_scores = f_scores / f_scores.max()
                    for feat, score in zip(features_df.columns, f_scores):
                        feature_scores[feat] = feature_scores.get(feat, 0) + score
                    selection_log.append("Computed ANOVA F-scores")
                except Exception as e:
                    selection_log.append(f"ANOVA scoring failed: {str(e)}")
        else:
            # No labels - use variance as importance
            for col in features_df.columns:
                feature_scores[col] = float(features_df[col].var())
            selection_log.append("Using variance as importance (no labels)")

        # Step 4: Select top N features
        if feature_scores:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in sorted_features[:n_features]]
        else:
            selected_features = list(features_df.columns[:n_features])

        # Compute feature importance relative scores
        max_score = max(feature_scores.values()) if feature_scores else 1.0
        importance_scores = {
            feat: float(score / max_score) if max_score > 0 else 0.0
            for feat, score in feature_scores.items()
        }

        # Store selection session
        selection_session_id = f"selection_{feature_session_id}"
        _selection_sessions[selection_session_id] = {
            'feature_session_id': feature_session_id,
            'selected_features': selected_features,
            'importance_scores': importance_scores,
            'removed_features': removed_features,
            'method': method,
            'original_count': len(original_features),
            'final_count': len(selected_features)
        }

        return {
            'session_id': selection_session_id,
            'selected_features': selected_features,
            'importance_scores': {f: importance_scores.get(f, 0) for f in selected_features},
            'all_scores': importance_scores,
            'removed_features': removed_features,
            'selection_log': selection_log,
            'original_count': len(original_features),
            'after_filtering': len(features_df.columns),
            'final_count': len(selected_features),
            'method': method
        }

    def get_feature_correlations(
        self,
        feature_session_id: str,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get correlation matrix for features (for visualization).

        Args:
            feature_session_id: Feature session ID
            features: Optional list of features to include

        Returns:
            Correlation matrix data
        """
        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features']

        if features:
            features_df = features_df[[f for f in features if f in features_df.columns]]

        corr_matrix = features_df.corr()

        return {
            'features': list(corr_matrix.columns),
            'correlations': corr_matrix.values.tolist(),
            'num_features': len(corr_matrix.columns)
        }

    def apply_selection(
        self,
        feature_session_id: str,
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """
        Apply feature selection and create a new reduced feature session.

        Args:
            feature_session_id: Original feature session ID
            selected_features: List of features to keep

        Returns:
            New feature session with reduced features
        """
        session = _feature_sessions.get(feature_session_id)
        if not session:
            raise ValueError(f"Feature session not found: {feature_session_id}")

        features_df = session['features']
        labels = session['labels']
        categories = session.get('categories')

        # Validate selected features exist
        valid_features = [f for f in selected_features if f in features_df.columns]
        if not valid_features:
            raise ValueError("No valid features selected")

        # Create reduced dataframe
        reduced_df = features_df[valid_features].copy()

        # Create new session
        new_session_id = f"selected_{feature_session_id}"
        _feature_sessions[new_session_id] = {
            'features': reduced_df,
            'labels': labels,
            'categories': categories,
            'feature_names': valid_features,
            'metadata': {
                **session.get('metadata', {}),
                'num_features': len(valid_features),
                'original_session': feature_session_id,
                'selection_applied': True
            }
        }

        return {
            'session_id': new_session_id,
            'num_features': len(valid_features),
            'selected_features': valid_features,
            'num_windows': len(reduced_df)
        }
