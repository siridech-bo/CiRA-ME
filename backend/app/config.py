"""
CiRA ME - Configuration Settings
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Security
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'cira-me-dev-secret-key-change-in-production')
    SESSION_LIFETIME_HOURS: int = 8

    # Paths
    DATASETS_ROOT_PATH: str = os.environ.get('DATASETS_ROOT_PATH', './datasets')
    SHARED_FOLDER_PATH: str = 'shared'
    DATABASE_PATH: str = './data/cirame.db'
    MODELS_PATH: str = './models'

    # Upload limits
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB

    # Roles
    ROLE_ADMIN: str = 'admin'
    ROLE_ANNOTATOR: str = 'annotator'

    # ML Settings
    DEFAULT_WINDOW_SIZE: int = 128
    DEFAULT_STRIDE: int = 64
    DEFAULT_CONTAMINATION: float = 0.1

    # Feature extraction
    TSFRESH_FEATURES: list = None
    CUSTOM_DSP_FEATURES: list = None

    def __post_init__(self):
        self.TSFRESH_FEATURES = [
            'mean', 'std', 'min', 'max', 'median',
            'sum', 'variance', 'skewness', 'kurtosis',
            'abs_energy', 'root_mean_square', 'mean_abs_change',
            'mean_change', 'length', 'sum_of_reoccurring_values',
            'sum_of_reoccurring_data_points', 'ratio_beyond_r_sigma',
            'count_above_mean', 'count_below_mean', 'longest_strike_above_mean',
            'longest_strike_below_mean', 'first_location_of_maximum',
            'first_location_of_minimum', 'last_location_of_maximum',
            'last_location_of_minimum', 'percentage_of_reoccurring_values'
        ]

        self.CUSTOM_DSP_FEATURES = [
            'rms', 'peak_to_peak', 'crest_factor', 'shape_factor',
            'impulse_factor', 'margin_factor', 'zero_crossing_rate',
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'spectral_flatness', 'spectral_entropy', 'peak_frequency',
            'band_power_low', 'band_power_mid', 'band_power_high'
        ]


# Anomaly detection algorithms (PyOD)
ANOMALY_ALGORITHMS = {
    'iforest': {'name': 'Isolation Forest', 'class': 'IForest'},
    'lof': {'name': 'Local Outlier Factor', 'class': 'LOF'},
    'ocsvm': {'name': 'One-Class SVM', 'class': 'OCSVM'},
    'hbos': {'name': 'Histogram-based Outlier Score', 'class': 'HBOS'},
    'knn': {'name': 'K-Nearest Neighbors', 'class': 'KNN'},
    'copod': {'name': 'COPOD', 'class': 'COPOD'},
    'ecod': {'name': 'ECOD', 'class': 'ECOD'},
    'suod': {'name': 'SUOD', 'class': 'SUOD'},
    'autoencoder': {'name': 'AutoEncoder', 'class': 'AutoEncoder'},
    'deep_svdd': {'name': 'Deep SVDD', 'class': 'DeepSVDD'},
}

# Classification algorithms (Scikit-learn)
CLASSIFICATION_ALGORITHMS = {
    'rf': {'name': 'Random Forest', 'class': 'RandomForestClassifier'},
    'gb': {'name': 'Gradient Boosting', 'class': 'GradientBoostingClassifier'},
    'svm': {'name': 'Support Vector Machine', 'class': 'SVC'},
    'mlp': {'name': 'Multi-Layer Perceptron', 'class': 'MLPClassifier'},
    'knn': {'name': 'K-Nearest Neighbors', 'class': 'KNeighborsClassifier'},
    'dt': {'name': 'Decision Tree', 'class': 'DecisionTreeClassifier'},
    'nb': {'name': 'Naive Bayes', 'class': 'GaussianNB'},
    'lr': {'name': 'Logistic Regression', 'class': 'LogisticRegression'},
}
