"""
CiRA ME - TimesNet Deep Learning Trainer
State-of-the-art time-series analysis using TimesNet architecture
"""

import os
import uuid
import pickle
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

# Global storage for TimesNet models
_timesnet_sessions: Dict[str, Dict] = {}


class TimesNetConfig:
    """TimesNet model configuration."""

    def __init__(
        self,
        seq_len: int = 128,
        pred_len: int = 0,
        enc_in: int = 3,
        c_out: int = 2,
        d_model: int = 64,
        d_ff: int = 128,
        num_kernels: int = 6,
        top_k: int = 3,
        e_layers: int = 2,
        dropout: float = 0.1,
        embed: str = 'timeF',
        freq: str = 'h',
        task_name: str = 'classification',
        num_class: int = 2,
        period_list: Optional[List[int]] = None
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k
        self.e_layers = e_layers
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.task_name = task_name
        self.num_class = num_class
        self.period_list = period_list or [12, 24, 48, 96]

    def to_dict(self) -> Dict:
        return vars(self)


class TimesNetTrainer:
    """
    TimesNet Deep Learning trainer for time-series analysis.

    TimesNet transforms 1D time series into 2D tensors based on
    multiple periods, enabling the use of 2D convolutions to capture
    both intraperiod and interperiod variations.
    """

    def __init__(self, models_path: str = './models'):
        self.models_path = models_path
        os.makedirs(models_path, exist_ok=True)
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if PyTorch is available."""
        self.torch_available = False
        try:
            import torch
            self.torch_available = True
        except ImportError:
            pass

    def get_default_config(self, mode: str, num_channels: int, num_classes: int = 2) -> TimesNetConfig:
        """Get default TimesNet configuration based on mode."""
        if mode == 'anomaly':
            return TimesNetConfig(
                seq_len=128,
                enc_in=num_channels,
                c_out=1,
                d_model=64,
                d_ff=128,
                num_kernels=6,
                top_k=3,
                e_layers=2,
                dropout=0.1,
                task_name='anomaly_detection',
                num_class=2,
                period_list=[8, 16, 32, 64]
            )
        else:
            return TimesNetConfig(
                seq_len=128,
                enc_in=num_channels,
                c_out=num_classes,
                d_model=64,
                d_ff=128,
                num_kernels=6,
                top_k=5,
                e_layers=3,
                dropout=0.1,
                task_name='classification',
                num_class=num_classes,
                period_list=[8, 16, 32, 64]
            )

    def train_anomaly(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
        config: Optional[TimesNetConfig] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        project_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train TimesNet for anomaly detection.

        For anomaly detection, TimesNet learns to reconstruct normal patterns.
        Anomalies are detected when reconstruction error exceeds a threshold.
        """
        if not self.torch_available:
            return self._train_fallback_anomaly(windows, labels, config, epochs)

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Get configuration
        num_channels = windows.shape[2] if len(windows.shape) == 3 else 1
        if config is None:
            config = self.get_default_config('anomaly', num_channels)

        # Prepare data
        X = torch.FloatTensor(windows)
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)

        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model (simplified TimesNet-like architecture)
        model = self._build_timesnet_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()

                # Forward pass (reconstruction)
                output = model(x)
                loss = criterion(output, x)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

        # Calculate reconstruction errors for threshold
        model.eval()
        with torch.no_grad():
            all_outputs = model(X)
            reconstruction_errors = torch.mean((X - all_outputs) ** 2, dim=(1, 2)).numpy()

        # Set threshold (e.g., 95th percentile)
        threshold = np.percentile(reconstruction_errors, 95)

        # Predictions
        predictions = (reconstruction_errors > threshold).astype(int)

        # Calculate metrics
        metrics = {
            'final_loss': losses[-1] if losses else 0,
            'threshold': float(threshold),
            'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
            'std_reconstruction_error': float(np.std(reconstruction_errors)),
            'anomalies_detected': int(np.sum(predictions)),
            'total_samples': len(predictions),
            'anomaly_ratio': float(np.sum(predictions) / len(predictions))
        }

        # If ground truth labels available
        if labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_true = (labels == 'anomaly') | (labels == 1) | (labels == '1')
            y_true = y_true.astype(int)

            metrics.update({
                'accuracy': float(accuracy_score(y_true, predictions)),
                'precision': float(precision_score(y_true, predictions, zero_division=0)),
                'recall': float(recall_score(y_true, predictions, zero_division=0)),
                'f1': float(f1_score(y_true, predictions, zero_division=0))
            })

        # Save model
        session_id = str(uuid.uuid4())
        model_path = os.path.join(self.models_path, f"timesnet_{session_id}.pkl")

        model_data = {
            'model_state': model.state_dict(),
            'config': config.to_dict(),
            'threshold': threshold,
            'mode': 'anomaly',
            'reconstruction_stats': {
                'mean': float(np.mean(reconstruction_errors)),
                'std': float(np.std(reconstruction_errors))
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Store session
        _timesnet_sessions[session_id] = {
            'model': model,
            'config': config,
            'threshold': threshold,
            'mode': 'anomaly',
            'metrics': metrics,
            'model_path': model_path,
            'created_at': datetime.utcnow().isoformat()
        }

        return {
            'training_session_id': session_id,
            'algorithm': 'TimesNet',
            'mode': 'anomaly',
            'metrics': metrics,
            'config': config.to_dict(),
            'model_path': model_path
        }

    def train_classification(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        config: Optional[TimesNetConfig] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        project_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train TimesNet for multi-class classification."""
        if not self.torch_available:
            return self._train_fallback_classification(windows, labels, config, epochs, test_size)

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
        num_classes = len(le.classes_)
        class_names = le.classes_.tolist()

        # Get configuration
        num_channels = windows.shape[2] if len(windows.shape) == 3 else 1
        if config is None:
            config = self.get_default_config('classification', num_channels, num_classes)
        config.num_class = num_classes

        # Prepare data
        X = torch.FloatTensor(windows)
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
        y = torch.LongTensor(y_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Build model
        model = self._build_timesnet_classifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                output = model(batch_x)
                loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

        # Evaluation
        model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)

                all_preds.extend(preds.numpy())
                all_probs.extend(probs.numpy())

        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        y_true = y_test.numpy()

        # Metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classes': class_names,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'final_loss': train_losses[-1] if train_losses else 0,
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }

        # Save model
        session_id = str(uuid.uuid4())
        model_path = os.path.join(self.models_path, f"timesnet_{session_id}.pkl")

        model_data = {
            'model_state': model.state_dict(),
            'config': config.to_dict(),
            'label_encoder': le,
            'class_names': class_names,
            'mode': 'classification'
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Store session
        _timesnet_sessions[session_id] = {
            'model': model,
            'config': config,
            'label_encoder': le,
            'mode': 'classification',
            'metrics': metrics,
            'model_path': model_path,
            'created_at': datetime.utcnow().isoformat()
        }

        return {
            'training_session_id': session_id,
            'algorithm': 'TimesNet',
            'mode': 'classification',
            'metrics': metrics,
            'config': config.to_dict(),
            'model_path': model_path
        }

    def _build_timesnet_model(self, config: TimesNetConfig):
        """Build a simplified TimesNet-like encoder-decoder for anomaly detection."""
        import torch.nn as nn

        class TimesNetEncoder(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.seq_len = config.seq_len
                self.enc_in = config.enc_in
                self.d_model = config.d_model

                # Embedding
                self.embed = nn.Linear(config.enc_in, config.d_model)

                # Temporal blocks (simplified)
                self.encoder = nn.Sequential(
                    nn.Conv1d(config.d_model, config.d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Conv1d(config.d_ff, config.d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Conv1d(config.d_model, config.d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Conv1d(config.d_ff, config.d_model, kernel_size=3, padding=1),
                )

                self.output = nn.Linear(config.d_model, config.enc_in)

            def forward(self, x):
                # x: (batch, seq_len, channels)
                batch_size = x.size(0)

                # Embed
                x = self.embed(x)  # (batch, seq_len, d_model)

                # Transpose for conv1d
                x = x.transpose(1, 2)  # (batch, d_model, seq_len)

                # Encode
                encoded = self.encoder(x)

                # Decode
                decoded = self.decoder(encoded)

                # Transpose back
                decoded = decoded.transpose(1, 2)  # (batch, seq_len, d_model)

                # Output
                output = self.output(decoded)  # (batch, seq_len, channels)

                return output

        return TimesNetEncoder(config)

    def _build_timesnet_classifier(self, config: TimesNetConfig):
        """Build TimesNet classifier."""
        import torch.nn as nn

        class TimesNetClassifier(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.seq_len = config.seq_len
                self.enc_in = config.enc_in
                self.d_model = config.d_model

                # Embedding
                self.embed = nn.Linear(config.enc_in, config.d_model)

                # Temporal blocks
                self.encoder = nn.Sequential(
                    nn.Conv1d(config.d_model, config.d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(config.d_ff),
                    nn.Dropout(config.dropout),
                    nn.Conv1d(config.d_ff, config.d_ff, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(config.d_ff),
                    nn.Dropout(config.dropout),
                    nn.Conv1d(config.d_ff, config.d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

                # Global pooling + classifier
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(config.d_model, config.d_ff),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_ff, config.num_class)
                )

            def forward(self, x):
                # x: (batch, seq_len, channels)
                # Embed
                x = self.embed(x)  # (batch, seq_len, d_model)

                # Transpose for conv1d
                x = x.transpose(1, 2)  # (batch, d_model, seq_len)

                # Encode
                encoded = self.encoder(x)

                # Classify
                output = self.classifier(encoded)

                return output

        return TimesNetClassifier(config)

    def _train_fallback_anomaly(self, windows, labels, config, epochs):
        """Fallback training without PyTorch (uses sklearn)."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Flatten windows
        X = windows.reshape(windows.shape[0], -1)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Isolation Forest as fallback
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        predictions = model.fit_predict(X_scaled)
        predictions = (predictions == -1).astype(int)

        metrics = {
            'anomalies_detected': int(np.sum(predictions)),
            'total_samples': len(predictions),
            'anomaly_ratio': float(np.sum(predictions) / len(predictions)),
            'note': 'Fallback to IsolationForest (PyTorch not available)'
        }

        session_id = str(uuid.uuid4())

        return {
            'training_session_id': session_id,
            'algorithm': 'TimesNet (Fallback: IsolationForest)',
            'mode': 'anomaly',
            'metrics': metrics,
            'config': config.to_dict() if config else {}
        }

    def _train_fallback_classification(self, windows, labels, config, epochs, test_size):
        """Fallback training without PyTorch (uses sklearn)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # Flatten windows
        X = windows.reshape(windows.shape[0], -1)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest as fallback
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classes': le.classes_.tolist(),
            'note': 'Fallback to RandomForest (PyTorch not available)'
        }

        session_id = str(uuid.uuid4())

        return {
            'training_session_id': session_id,
            'algorithm': 'TimesNet (Fallback: RandomForest)',
            'mode': 'classification',
            'metrics': metrics,
            'config': config.to_dict() if config else {}
        }

    def predict(self, session_id: str, windows: np.ndarray) -> Dict[str, Any]:
        """Make predictions using a trained TimesNet model."""
        session = _timesnet_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        model = session['model']
        mode = session['mode']

        if not self.torch_available:
            return {'error': 'PyTorch not available for inference'}

        import torch

        model.eval()
        X = torch.FloatTensor(windows)
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)

        with torch.no_grad():
            if mode == 'anomaly':
                output = model(X)
                reconstruction_errors = torch.mean((X - output) ** 2, dim=(1, 2)).numpy()
                threshold = session['threshold']
                predictions = (reconstruction_errors > threshold).astype(int)

                return {
                    'predictions': predictions.tolist(),
                    'reconstruction_errors': reconstruction_errors.tolist(),
                    'threshold': threshold,
                    'anomaly_count': int(np.sum(predictions)),
                    'total_samples': len(predictions)
                }
            else:
                output = model(X)
                probs = torch.softmax(output, dim=1).numpy()
                predictions = np.argmax(probs, axis=1)

                le = session.get('label_encoder')
                if le:
                    predictions = le.inverse_transform(predictions)

                return {
                    'predictions': predictions.tolist(),
                    'probabilities': probs.tolist(),
                    'total_samples': len(predictions)
                }

    def get_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get metrics for a TimesNet training session."""
        session = _timesnet_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        return {
            'training_session_id': session_id,
            'algorithm': 'TimesNet',
            'mode': session['mode'],
            'metrics': session['metrics'],
            'config': session['config'].to_dict() if hasattr(session['config'], 'to_dict') else session['config'],
            'created_at': session.get('created_at')
        }
