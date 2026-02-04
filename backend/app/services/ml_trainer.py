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
    confusion_matrix, classification_report, roc_auc_score
)

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
            if y_test is not None and len(np.unique(y_test)) == 2:
                y_binary = (y_test == 'anomaly') | (y_test == 1) | (y_test == '1')
                y_binary = y_binary.astype(int)

                metrics.update({
                    'accuracy': float(accuracy_score(y_binary, y_pred_test)),
                    'precision': float(precision_score(y_binary, y_pred_test, zero_division=0)),
                    'recall': float(recall_score(y_binary, y_pred_test, zero_division=0)),
                    'f1': float(f1_score(y_binary, y_pred_test, zero_division=0)),
                    'confusion_matrix': confusion_matrix(y_binary, y_pred_test).tolist()
                })

                try:
                    metrics['auc_roc'] = float(roc_auc_score(y_binary, decision_scores_test))
                except ValueError:
                    pass

        elif y_train is not None and len(np.unique(y_train)) == 2:
            # Fallback: evaluate on training data if no test set
            y_binary = (y_train == 'anomaly') | (y_train == 1) | (y_train == '1')
            y_binary = y_binary.astype(int)

            metrics.update({
                'accuracy': float(accuracy_score(y_binary, y_pred_train)),
                'precision': float(precision_score(y_binary, y_pred_train, zero_division=0)),
                'recall': float(recall_score(y_binary, y_pred_train, zero_division=0)),
                'f1': float(f1_score(y_binary, y_pred_train, zero_division=0)),
                'confusion_matrix': confusion_matrix(y_binary, y_pred_train).tolist()
            })

            try:
                metrics['auc_roc'] = float(roc_auc_score(y_binary, decision_scores_train))
            except ValueError:
                pass

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

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classes': classes.tolist() if hasattr(classes, 'tolist') else list(classes),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'split_method': split_method
        }

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report

        # AUC-ROC for binary classification
        if is_binary and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                metrics['auc_roc'] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                pass

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
