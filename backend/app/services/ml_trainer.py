"""
CiRA ME - ML Training Service
Handles Anomaly Detection (PyOD) and Classification (Scikit-learn)
"""

import os
import uuid
import pickle
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

from .feature_extractor import FeatureExtractor, _feature_sessions

# Global storage for trained models
_model_sessions: Dict[str, Dict] = {}


class MLTrainer:
    """Service for training ML models."""

    def __init__(self, models_path: str = './models'):
        self.models_path = models_path
        os.makedirs(models_path, exist_ok=True)

    def _get_pyod_model(self, algorithm: str, hyperparameters: Dict):
        """Get PyOD model instance."""
        try:
            if algorithm == 'iforest':
                from pyod.models.iforest import IForest
                return IForest(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    contamination=hyperparameters.get('contamination', 0.1),
                    max_features=hyperparameters.get('max_features', 1.0),
                    random_state=42
                )
            elif algorithm == 'lof':
                from pyod.models.lof import LOF
                return LOF(
                    n_neighbors=hyperparameters.get('n_neighbors', 20),
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            elif algorithm == 'ocsvm':
                from pyod.models.ocsvm import OCSVM
                return OCSVM(
                    kernel=hyperparameters.get('kernel', 'rbf'),
                    nu=hyperparameters.get('nu', 0.1)
                )
            elif algorithm == 'hbos':
                from pyod.models.hbos import HBOS
                return HBOS(
                    n_bins=hyperparameters.get('n_bins', 10),
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            elif algorithm == 'knn':
                from pyod.models.knn import KNN
                return KNN(
                    n_neighbors=hyperparameters.get('n_neighbors', 5),
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            elif algorithm == 'copod':
                from pyod.models.copod import COPOD
                return COPOD(
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            elif algorithm == 'ecod':
                from pyod.models.ecod import ECOD
                return ECOD(
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            elif algorithm == 'suod':
                from pyod.models.suod import SUOD
                return SUOD(
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            elif algorithm == 'autoencoder':
                from pyod.models.auto_encoder import AutoEncoder
                return AutoEncoder(
                    hidden_neurons=hyperparameters.get('hidden_neurons', [64, 32, 32, 64]),
                    contamination=hyperparameters.get('contamination', 0.1),
                    epochs=hyperparameters.get('epochs', 100)
                )
            elif algorithm == 'deep_svdd':
                from pyod.models.deep_svdd import DeepSVDD
                return DeepSVDD(
                    n_features=hyperparameters.get('n_features', 10),
                    contamination=hyperparameters.get('contamination', 0.1)
                )
            else:
                raise ValueError(f"Unknown PyOD algorithm: {algorithm}")
        except ImportError as e:
            raise ImportError(f"PyOD library required. Install with: pip install pyod. Error: {e}")

    def _get_sklearn_model(self, algorithm: str, hyperparameters: Dict):
        """Get Scikit-learn model instance."""
        if algorithm == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth', None),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                random_state=42
            )
        elif algorithm == 'gb':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                learning_rate=hyperparameters.get('learning_rate', 0.1),
                max_depth=hyperparameters.get('max_depth', 3),
                random_state=42
            )
        elif algorithm == 'svm':
            from sklearn.svm import SVC
            return SVC(
                kernel=hyperparameters.get('kernel', 'rbf'),
                C=hyperparameters.get('C', 1.0),
                gamma=hyperparameters.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        elif algorithm == 'mlp':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=hyperparameters.get('hidden_layer_sizes', (100,)),
                activation=hyperparameters.get('activation', 'relu'),
                max_iter=hyperparameters.get('max_iter', 500),
                random_state=42
            )
        elif algorithm == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(
                n_neighbors=hyperparameters.get('n_neighbors', 5),
                weights=hyperparameters.get('weights', 'uniform')
            )
        elif algorithm == 'dt':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                max_depth=hyperparameters.get('max_depth', None),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                random_state=42
            )
        elif algorithm == 'nb':
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        elif algorithm == 'lr':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=hyperparameters.get('C', 1.0),
                max_iter=hyperparameters.get('max_iter', 1000),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown Scikit-learn algorithm: {algorithm}")

    def train_anomaly(
        self,
        feature_session_id: str,
        algorithm: str,
        hyperparameters: Dict = None,
        project_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train an anomaly detection model using proper category-based splitting."""
        hyperparameters = hyperparameters or {}

        # Get feature data with categories
        X, y, categories = FeatureExtractor.get_features_for_training(feature_session_id)

        # Use category-based split if available
        split_method = 'category'
        if categories is not None:
            train_mask = np.isin(categories, ['training', 'train'])
            test_mask = np.isin(categories, ['testing', 'test'])

            if np.sum(train_mask) > 0:
                X_train = X[train_mask]
                X_test = X[test_mask] if np.sum(test_mask) > 0 else None
                y_train = y[train_mask] if y is not None else None
                y_test = y[test_mask] if y is not None and np.sum(test_mask) > 0 else None
            else:
                # No training category found, use all data
                X_train = X
                X_test = None
                y_train = y
                y_test = None
                split_method = 'all_data'
        else:
            X_train = X
            X_test = None
            y_train = y
            y_test = None
            split_method = 'all_data'

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Get model
        model = self._get_pyod_model(algorithm, hyperparameters)

        # Train on training data only
        model.fit(X_train_scaled)

        # Get predictions and scores on training data
        y_pred_train = model.labels_
        decision_scores_train = model.decision_scores_

        # Calculate metrics
        metrics = {
            'contamination': hyperparameters.get('contamination', 0.1),
            'anomalies_detected': int(np.sum(y_pred_train)),
            'total_samples': len(y_pred_train),
            'anomaly_ratio': float(np.sum(y_pred_train) / len(y_pred_train)),
            'decision_score_mean': float(np.mean(decision_scores_train)),
            'decision_score_std': float(np.std(decision_scores_train)),
            'threshold': float(model.threshold_),
            'split_method': split_method,
            'train_samples': len(X_train)
        }

        # Evaluate on test set if available
        if X_test is not None and len(X_test) > 0:
            X_test_scaled = scaler.transform(X_test)
            y_pred_test = model.predict(X_test_scaled)
            decision_scores_test = model.decision_function(X_test_scaled)
            metrics['test_samples'] = len(X_test)
            metrics['test_anomalies_detected'] = int(np.sum(y_pred_test))

            # If we have ground truth test labels
            if y_test is not None and len(y_test) > 0:
                # Convert labels to binary (0=normal, 1=anomaly)
                # Handle various label formats
                y_test_arr = np.array(y_test)
                if y_test_arr.dtype == np.object_ or y_test_arr.dtype.kind in ['U', 'S']:
                    # String labels - check for anomaly indicators
                    y_binary = np.array([
                        1 if str(l).lower() in ['anomaly', 'abnormal', 'fault', 'failure', 'attack', 'malicious', '1', 'true', 'yes', 'bad']
                        else 0
                        for l in y_test_arr
                    ])
                else:
                    # Numeric labels - assume non-zero is anomaly
                    y_binary = (y_test_arr != 0).astype(int)

                # Only calculate metrics if we have both classes
                if len(np.unique(y_binary)) == 2:
                    # Basic metrics
                    metrics.update({
                        'accuracy': float(accuracy_score(y_binary, y_pred_test)),
                        'precision': float(precision_score(y_binary, y_pred_test, zero_division=0)),
                        'recall': float(recall_score(y_binary, y_pred_test, zero_division=0)),
                        'f1': float(f1_score(y_binary, y_pred_test, zero_division=0))
                    })

                    # Confusion matrix with labels
                    cm = confusion_matrix(y_binary, y_pred_test)
                    metrics['confusion_matrix'] = cm.tolist()
                    metrics['confusion_matrix_labels'] = ['Normal', 'Anomaly']
                    metrics['classes'] = ['Normal', 'Anomaly']

                    # TP/TN/FP/FN breakdown
                    tn, fp, fn, tp = cm.ravel()
                    metrics['tp'] = int(tp)
                    metrics['tn'] = int(tn)
                    metrics['fp'] = int(fp)
                    metrics['fn'] = int(fn)
                    # Also use long names for frontend compatibility
                    metrics['true_positives'] = int(tp)
                    metrics['true_negatives'] = int(tn)
                    metrics['false_positives'] = int(fp)
                    metrics['false_negatives'] = int(fn)

                    # Per-class metrics
                    metrics['per_class_metrics'] = [
                        {
                            'class': 'Normal',
                            'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
                            'recall': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
                            'support': int(tn + fp)
                        },
                        {
                            'class': 'Anomaly',
                            'precision': float(precision_score(y_binary, y_pred_test, zero_division=0)),
                            'recall': float(recall_score(y_binary, y_pred_test, zero_division=0)),
                            'support': int(tp + fn)
                        }
                    ]

                    # ROC curve
                    try:
                        fpr, tpr, _ = roc_curve(y_binary, decision_scores_test)
                        metrics['auc_roc'] = float(roc_auc_score(y_binary, decision_scores_test))
                        metrics['roc_auc'] = metrics['auc_roc']  # Alias for frontend
                        # Downsample curve points (max 100 points)
                        if len(fpr) > 100:
                            indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
                            fpr = fpr[indices]
                            tpr = tpr[indices]
                        metrics['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist()
                        }
                    except ValueError:
                        pass

                    # PR curve
                    try:
                        precision_curve, recall_curve, _ = precision_recall_curve(y_binary, decision_scores_test)
                        metrics['auc_pr'] = float(average_precision_score(y_binary, decision_scores_test))
                        # Downsample curve points
                        if len(precision_curve) > 100:
                            indices = np.linspace(0, len(precision_curve) - 1, 100, dtype=int)
                            precision_curve = precision_curve[indices]
                            recall_curve = recall_curve[indices]
                        metrics['pr_curve'] = {
                            'precision': precision_curve.tolist(),
                            'recall': recall_curve.tolist()
                        }
                    except ValueError:
                        pass
                else:
                    # Test labels don't have both classes
                    unique_labels = np.unique(y_binary)
                    metrics['metrics_info'] = f'Cannot calculate accuracy metrics - test data has only {len(unique_labels)} class(es). Need both normal and anomaly samples.'

        elif y_train is not None and len(y_train) > 0:
            # Fallback: evaluate on training data if no test set
            # Convert labels to binary (0=normal, 1=anomaly)
            y_train_arr = np.array(y_train)
            if y_train_arr.dtype == np.object_ or y_train_arr.dtype.kind in ['U', 'S']:
                # String labels - check for anomaly indicators
                y_binary = np.array([
                    1 if str(l).lower() in ['anomaly', 'abnormal', 'fault', 'failure', 'attack', 'malicious', '1', 'true', 'yes', 'bad']
                    else 0
                    for l in y_train_arr
                ])
            else:
                # Numeric labels - assume non-zero is anomaly
                y_binary = (y_train_arr != 0).astype(int)

            # Only calculate if we have both classes
            if len(np.unique(y_binary)) == 2:
                # Basic metrics
                metrics.update({
                    'accuracy': float(accuracy_score(y_binary, y_pred_train)),
                    'precision': float(precision_score(y_binary, y_pred_train, zero_division=0)),
                    'recall': float(recall_score(y_binary, y_pred_train, zero_division=0)),
                    'f1': float(f1_score(y_binary, y_pred_train, zero_division=0))
                })

                # Confusion matrix with labels
                cm = confusion_matrix(y_binary, y_pred_train)
                metrics['confusion_matrix'] = cm.tolist()
                metrics['confusion_matrix_labels'] = ['Normal', 'Anomaly']
                metrics['classes'] = ['Normal', 'Anomaly']

                # TP/TN/FP/FN breakdown
                tn, fp, fn, tp = cm.ravel()
                metrics['tp'] = int(tp)
                metrics['tn'] = int(tn)
                metrics['fp'] = int(fp)
                metrics['fn'] = int(fn)
                # Also use long names for frontend compatibility
                metrics['true_positives'] = int(tp)
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)

                # Per-class metrics
                metrics['per_class_metrics'] = [
                    {
                        'class': 'Normal',
                        'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
                        'recall': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
                        'support': int(tn + fp)
                    },
                    {
                        'class': 'Anomaly',
                        'precision': float(precision_score(y_binary, y_pred_train, zero_division=0)),
                        'recall': float(recall_score(y_binary, y_pred_train, zero_division=0)),
                        'support': int(tp + fn)
                    }
                ]

                # ROC curve
                try:
                    fpr, tpr, _ = roc_curve(y_binary, decision_scores_train)
                    metrics['auc_roc'] = float(roc_auc_score(y_binary, decision_scores_train))
                    metrics['roc_auc'] = metrics['auc_roc']  # Alias for frontend
                    # Downsample curve points
                    if len(fpr) > 100:
                        indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
                        fpr = fpr[indices]
                        tpr = tpr[indices]
                    metrics['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    }
                except ValueError:
                    pass

                # PR curve
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_binary, decision_scores_train)
                    metrics['auc_pr'] = float(average_precision_score(y_binary, decision_scores_train))
                    # Downsample curve points
                    if len(precision_curve) > 100:
                        indices = np.linspace(0, len(precision_curve) - 1, 100, dtype=int)
                        precision_curve = precision_curve[indices]
                        recall_curve = recall_curve[indices]
                    metrics['pr_curve'] = {
                        'precision': precision_curve.tolist(),
                        'recall': recall_curve.tolist()
                    }
                except ValueError:
                    pass
            else:
                # Labels don't have both classes - can't calculate supervised metrics
                unique_labels = np.unique(y_binary)
                metrics['metrics_info'] = f'Cannot calculate accuracy metrics - training data has only {len(unique_labels)} class(es). Anomaly detection requires labeled test data with both normal and anomaly samples.'

        # If no labels available at all
        if y_train is None or len(y_train) == 0:
            if X_test is None:
                metrics['metrics_info'] = 'No labeled data available. Model trained using unsupervised anomaly detection. To see accuracy metrics, provide labeled test data.'

        # Generate session ID and save model
        training_session_id = str(uuid.uuid4())
        model_path = os.path.join(self.models_path, f"{training_session_id}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'algorithm': algorithm,
                'mode': 'anomaly',
                'hyperparameters': hyperparameters,
                'feature_session_id': feature_session_id
            }, f)

        # Store in session
        _model_sessions[training_session_id] = {
            'model': model,
            'scaler': scaler,
            'algorithm': algorithm,
            'mode': 'anomaly',
            'metrics': metrics,
            'model_path': model_path,
            'hyperparameters': hyperparameters,
            'created_at': datetime.utcnow().isoformat()
        }

        return {
            'training_session_id': training_session_id,
            'algorithm': algorithm,
            'mode': 'anomaly',
            'metrics': metrics,
            'model_path': model_path
        }

    def train_classification(
        self,
        feature_session_id: str,
        algorithm: str,
        hyperparameters: Dict = None,
        test_size: float = 0.2,
        project_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train a classification model using proper category-based splitting.

        If the dataset has a 'category' column (training/testing), uses that
        for the train/test split instead of random splitting. This prevents
        data leakage from overlapping windows.
        """
        hyperparameters = hyperparameters or {}

        # Get feature data with categories
        X, y, categories = FeatureExtractor.get_features_for_training(feature_session_id)

        if y is None:
            raise ValueError("Labels required for classification training")

        # Use category-based split if available
        split_method = 'category'
        if categories is not None:
            train_mask = np.isin(categories, ['training', 'train'])
            test_mask = np.isin(categories, ['testing', 'test'])

            if np.sum(train_mask) > 0 and np.sum(test_mask) > 0:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
            else:
                # Fallback to random split if categories don't have both train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                split_method = 'random'
        else:
            # No categories - use random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            split_method = 'random'

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Get model
        model = self._get_sklearn_model(algorithm, hyperparameters)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        classes = np.unique(y)
        is_binary = len(classes) == 2

        # Basic metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'split_method': split_method
        }

        # Convert classes to list safely
        class_list = classes.tolist() if hasattr(classes, 'tolist') else list(classes)
        class_labels = [str(c) for c in class_list]
        metrics['classes'] = class_list

        # Confusion matrix with labels
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['confusion_matrix_labels'] = class_labels

        # TP/TN/FP/FN breakdown (for binary classification or per-class)
        if is_binary:
            # Binary case - direct TP/TN/FP/FN
            tn, fp, fn, tp = cm.ravel()
            metrics['tp'] = int(tp)
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            # Also use long names for frontend compatibility
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
        else:
            # Multiclass - compute per class
            per_class_metrics = []
            for i, cls in enumerate(class_labels):
                # One-vs-rest approach
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp
                per_class_metrics.append({
                    'class': cls,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                })
            metrics['per_class_breakdown'] = per_class_metrics
            # Also provide totals
            metrics['tp'] = int(np.trace(cm))
            metrics['fn'] = int(cm.sum() - np.trace(cm))

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report

        # Per-class metrics for display
        per_class = []
        for cls in class_list:
            cls_str = str(cls)
            if cls_str in report:
                per_class.append({
                    'class': cls_str,
                    'precision': float(report[cls_str].get('precision', 0)),
                    'recall': float(report[cls_str].get('recall', 0)),
                    'f1': float(report[cls_str].get('f1-score', 0)),
                    'support': int(report[cls_str].get('support', 0))
                })
        metrics['per_class_metrics'] = per_class

        # ROC and PR curves
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test_scaled)

                if is_binary:
                    # Binary classification ROC curve
                    y_proba_pos = y_proba[:, 1]
                    # Convert labels to binary
                    y_test_binary = (y_test == classes[1]).astype(int)

                    # ROC curve
                    fpr, tpr, thresholds = roc_curve(y_test_binary, y_proba_pos)
                    metrics['auc_roc'] = float(roc_auc_score(y_test_binary, y_proba_pos))
                    metrics['roc_auc'] = metrics['auc_roc']  # Alias for frontend
                    # Downsample curve points for UI (max 100 points)
                    if len(fpr) > 100:
                        indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
                        fpr = fpr[indices]
                        tpr = tpr[indices]
                    metrics['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    }

                    # PR curve
                    precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_proba_pos)
                    metrics['auc_pr'] = float(average_precision_score(y_test_binary, y_proba_pos))
                    # Downsample curve points
                    if len(precision_curve) > 100:
                        indices = np.linspace(0, len(precision_curve) - 1, 100, dtype=int)
                        precision_curve = precision_curve[indices]
                        recall_curve = recall_curve[indices]
                    metrics['pr_curve'] = {
                        'precision': precision_curve.tolist(),
                        'recall': recall_curve.tolist()
                    }
                else:
                    # Multiclass - compute macro-averaged ROC
                    # CRITICAL: Use model.classes_ for binarization to match y_proba column order
                    # model.predict_proba returns probabilities in order of model.classes_, NOT np.unique(y)
                    model_classes = model.classes_.tolist() if hasattr(model.classes_, 'tolist') else list(model.classes_)

                    # Debug: Check for class ordering mismatch
                    print(f"[ML DEBUG] class_list (from np.unique): {class_list}")
                    print(f"[ML DEBUG] model.classes_: {model_classes}")
                    print(f"[ML DEBUG] y_proba shape: {y_proba.shape}")
                    print(f"[ML DEBUG] y_proba row sums (should be ~1): {y_proba.sum(axis=1)[:5]}")

                    # Binarize using MODEL's class order (not sorted unique)
                    y_test_bin = label_binarize(y_test, classes=model_classes)
                    if y_test_bin.shape[1] == 1:
                        # This can happen with 2 classes
                        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

                    print(f"[ML DEBUG] y_test_bin shape: {y_test_bin.shape}")
                    print(f"[ML DEBUG] y_test unique values: {np.unique(y_test)}")

                    # Compute ROC curve per class and average
                    all_fpr = np.linspace(0, 1, 100)
                    mean_tpr = np.zeros_like(all_fpr)
                    valid_classes = 0

                    for i in range(len(model_classes)):
                        if i < y_proba.shape[1] and i < y_test_bin.shape[1]:
                            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                            class_auc = auc(fpr_i, tpr_i)
                            print(f"[ML DEBUG] Class {i} ({model_classes[i]}): AUC={class_auc:.4f}, samples={y_test_bin[:, i].sum()}")
                            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
                            valid_classes += 1

                    if valid_classes > 0:
                        mean_tpr /= valid_classes

                    print(f"[ML DEBUG] mean_tpr range: [{mean_tpr.min():.4f}, {mean_tpr.max():.4f}]")

                    # Print sample curve points to verify shape
                    sample_indices = [0, 25, 50, 75, 99]
                    print(f"[ML DEBUG] Sample ROC points (fpr, tpr):")
                    for idx in sample_indices:
                        print(f"  [{idx}] ({all_fpr[idx]:.3f}, {mean_tpr[idx]:.3f})")

                    metrics['roc_curve'] = {
                        'fpr': all_fpr.tolist(),
                        'tpr': mean_tpr.tolist()
                    }

                    try:
                        # IMPORTANT: Pass labels parameter so sklearn knows y_proba column order
                        metrics['auc_roc'] = float(roc_auc_score(
                            y_test, y_proba,
                            multi_class='ovr',
                            average='weighted',
                            labels=model_classes  # Tell sklearn the column order matches model.classes_
                        ))
                        metrics['roc_auc'] = metrics['auc_roc']  # Alias for frontend
                        print(f"[ML DEBUG] Final ROC-AUC (sklearn ovr weighted): {metrics['auc_roc']:.4f}")
                    except ValueError as e:
                        print(f"[ML DEBUG] sklearn roc_auc_score failed: {e}")
                        # Fallback: compute from the averaged curve
                        metrics['roc_auc'] = float(auc(all_fpr, mean_tpr))
                        metrics['auc_roc'] = metrics['roc_auc']
                        print(f"[ML DEBUG] Fallback ROC-AUC from curve: {metrics['roc_auc']:.4f}")

            except Exception as e:
                # Log but don't fail if curve computation fails
                print(f"Warning: Could not compute ROC/PR curves: {e}")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_session = _feature_sessions.get(feature_session_id)
            if feature_session:
                feature_names = feature_session.get('feature_names', [])
                importances = model.feature_importances_
                metrics['feature_importance'] = dict(zip(feature_names, importances.tolist()))

        # Generate session ID and save model
        training_session_id = str(uuid.uuid4())
        model_path = os.path.join(self.models_path, f"{training_session_id}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'algorithm': algorithm,
                'mode': 'classification',
                'hyperparameters': hyperparameters,
                'classes': classes.tolist() if hasattr(classes, 'tolist') else list(classes),
                'feature_session_id': feature_session_id
            }, f)

        # Store in session
        _model_sessions[training_session_id] = {
            'model': model,
            'scaler': scaler,
            'algorithm': algorithm,
            'mode': 'classification',
            'metrics': metrics,
            'model_path': model_path,
            'hyperparameters': hyperparameters,
            'classes': classes.tolist() if hasattr(classes, 'tolist') else list(classes),
            'created_at': datetime.utcnow().isoformat()
        }

        return {
            'training_session_id': training_session_id,
            'algorithm': algorithm,
            'mode': 'classification',
            'metrics': metrics,
            'model_path': model_path
        }

    def predict(self, training_session_id: str, feature_session_id: str) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        session = _model_sessions.get(training_session_id)
        if not session:
            raise ValueError(f"Training session not found: {training_session_id}")

        model = session['model']
        scaler = session['scaler']
        mode = session['mode']

        # Get feature data
        X, _, _ = FeatureExtractor.get_features_for_training(feature_session_id)
        X_scaled = scaler.transform(X)

        if mode == 'anomaly':
            predictions = model.predict(X_scaled)
            scores = model.decision_function(X_scaled)
            return {
                'predictions': predictions.tolist(),
                'decision_scores': scores.tolist(),
                'anomaly_count': int(np.sum(predictions)),
                'total_samples': len(predictions)
            }
        else:
            predictions = model.predict(X_scaled)
            result = {
                'predictions': predictions.tolist(),
                'total_samples': len(predictions)
            }

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                result['probabilities'] = probabilities.tolist()

            return result

    def get_metrics(self, training_session_id: str) -> Dict[str, Any]:
        """Get metrics for a training session."""
        session = _model_sessions.get(training_session_id)
        if not session:
            raise ValueError(f"Training session not found: {training_session_id}")

        return {
            'training_session_id': training_session_id,
            'algorithm': session['algorithm'],
            'mode': session['mode'],
            'metrics': session['metrics'],
            'hyperparameters': session.get('hyperparameters', {}),
            'created_at': session.get('created_at')
        }

    def export_model(self, training_session_id: str, export_format: str = 'pickle') -> Dict[str, Any]:
        """Export a trained model."""
        session = _model_sessions.get(training_session_id)
        if not session:
            raise ValueError(f"Training session not found: {training_session_id}")

        model_path = session['model_path']

        if export_format == 'pickle':
            return {
                'format': 'pickle',
                'path': model_path,
                'message': 'Model already saved as pickle'
            }
        elif export_format == 'joblib':
            import joblib
            joblib_path = model_path.replace('.pkl', '.joblib')
            joblib.dump({
                'model': session['model'],
                'scaler': session['scaler']
            }, joblib_path)
            return {
                'format': 'joblib',
                'path': joblib_path,
                'message': 'Model exported as joblib'
            }
        elif export_format == 'onnx':
            # ONNX export requires skl2onnx
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                model = session['model']
                feature_session = _feature_sessions.get(session.get('feature_session_id', ''))
                n_features = len(feature_session.get('feature_names', [])) if feature_session else 10

                initial_type = [('float_input', FloatTensorType([None, n_features]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)

                onnx_path = model_path.replace('.pkl', '.onnx')
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())

                return {
                    'format': 'onnx',
                    'path': onnx_path,
                    'message': 'Model exported as ONNX'
                }
            except ImportError:
                raise ImportError("skl2onnx required for ONNX export. Install with: pip install skl2onnx")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
