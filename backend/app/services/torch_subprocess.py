#!/usr/bin/env python3
"""
CiRA ME - Isolated PyTorch Training Subprocess

This script runs PyTorch training in a separate process to avoid DLL conflicts
with other CUDA applications (like SAM2) running on the same system.

Usage:
    python torch_subprocess.py <config_file> <output_file>
"""

import os
import sys
import json
import pickle
import traceback
import numpy as np
from datetime import datetime

# Set environment variables before importing torch
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def train_timesnet_anomaly(config: dict, data: dict) -> dict:
    """Train TimesNet for anomaly detection."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Get and validate device
    requested_device = config.get('device', 'cpu')
    print(f"[TimesNet Subprocess] Requested device: {requested_device}", file=sys.stderr)
    print(f"[TimesNet Subprocess] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)

    if requested_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[TimesNet Subprocess] Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    else:
        device = torch.device('cpu')
        if requested_device == 'cuda':
            print(f"[TimesNet Subprocess] CUDA requested but not available, using CPU", file=sys.stderr)

    print(f"[TimesNet Subprocess] Training on device: {device}", file=sys.stderr)

    # Load data
    windows = np.array(data['windows'])
    labels = data.get('labels')

    # Prepare data
    X = torch.FloatTensor(windows)
    if len(X.shape) == 2:
        X = X.unsqueeze(-1)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Build model
    model = build_timesnet_encoder(config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    losses = []

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{config['epochs']}] Loss: {avg_loss:.6f}", file=sys.stderr)

    # Calculate reconstruction errors
    model.eval()
    X_device = X.to(device)
    with torch.no_grad():
        all_outputs = model(X_device)
        reconstruction_errors = torch.mean((X_device - all_outputs) ** 2, dim=(1, 2)).cpu().numpy()

    # Parse labels if available to determine actual anomaly rate
    y_true = None
    actual_anomaly_rate = 0.1  # Default 10%

    if labels is not None:
        labels_arr = np.array(labels)
        # Handle various label formats
        if labels_arr.dtype == np.object_ or labels_arr.dtype.kind == 'U':
            # String labels
            y_true = np.array([
                1 if str(l).lower() in ['anomaly', 'abnormal', '1', 'true', 'yes', 'fault', 'failure']
                else 0
                for l in labels_arr
            ])
        else:
            # Numeric labels - assume non-zero is anomaly
            y_true = (labels_arr != 0).astype(int)

        actual_anomaly_rate = float(np.mean(y_true))
        print(f"[TimesNet] Detected anomaly rate in data: {actual_anomaly_rate:.2%}", file=sys.stderr)

    # Set threshold based on actual anomaly rate (or default)
    # Use percentile that matches expected anomaly rate
    threshold_percentile = (1 - actual_anomaly_rate) * 100
    threshold = float(np.percentile(reconstruction_errors, threshold_percentile))
    predictions = (reconstruction_errors > threshold).astype(int)

    print(f"[TimesNet] Threshold percentile: {threshold_percentile:.1f}%, threshold value: {threshold:.6f}", file=sys.stderr)
    print(f"[TimesNet] Predictions: {np.sum(predictions)} anomalies out of {len(predictions)} samples", file=sys.stderr)

    # Metrics
    metrics = {
        'final_loss': float(losses[-1]) if losses else 0,
        'threshold': threshold,
        'threshold_percentile': threshold_percentile,
        'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
        'std_reconstruction_error': float(np.std(reconstruction_errors)),
        'anomalies_detected': int(np.sum(predictions)),
        'total_samples': len(predictions),
        'anomaly_ratio': float(np.sum(predictions) / len(predictions))
    }

    # Calculate metrics against ground truth if available
    if y_true is not None:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_curve, auc, precision_recall_curve,
            average_precision_score
        )

        # Basic metrics
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # ROC curve data (using reconstruction errors as scores)
        fpr, tpr, roc_thresholds = roc_curve(y_true, reconstruction_errors)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, reconstruction_errors)
        pr_auc = average_precision_score(y_true, reconstruction_errors)

        # Sample ROC curve points (reduce to ~50 points for JSON efficiency)
        roc_sample_indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
        roc_data = {
            'fpr': [float(fpr[i]) for i in roc_sample_indices],
            'tpr': [float(tpr[i]) for i in roc_sample_indices],
            'auc': float(roc_auc)
        }

        # Sample PR curve points
        pr_sample_indices = np.linspace(0, len(precision_curve) - 1, min(50, len(precision_curve)), dtype=int)
        pr_data = {
            'precision': [float(precision_curve[i]) for i in pr_sample_indices],
            'recall': [float(recall_curve[i]) for i in pr_sample_indices],
            'auc': float(pr_auc)
        }

        metrics.update({
            'accuracy': float(accuracy_score(y_true, predictions)),
            'precision': float(precision_score(y_true, predictions, zero_division=0)),
            'recall': float(recall_score(y_true, predictions, zero_division=0)),
            'f1': float(f1_score(y_true, predictions, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': ['Normal', 'Anomaly'],
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            'true_anomaly_rate': actual_anomaly_rate,
            'roc_curve': roc_data,
            'pr_curve': pr_data,
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        })

        print(f"[TimesNet] Metrics - Acc: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}, "
              f"Precision: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}, "
              f"ROC-AUC: {roc_auc:.3f}", file=sys.stderr)

    # Save model state
    model_state = {
        'model_state_dict': {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
        'config': config,
        'threshold': threshold,
        'mode': 'anomaly'
    }

    return {
        'success': True,
        'metrics': metrics,
        'model_state': model_state,
        'device': str(device)
    }


def train_timesnet_classification(config: dict, data: dict) -> dict:
    """Train TimesNet for classification."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Get and validate device
    requested_device = config.get('device', 'cpu')
    print(f"[TimesNet Subprocess] Requested device: {requested_device}", file=sys.stderr)
    print(f"[TimesNet Subprocess] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)

    if requested_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[TimesNet Subprocess] Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    else:
        device = torch.device('cpu')
        if requested_device == 'cuda':
            print(f"[TimesNet Subprocess] CUDA requested but not available, using CPU", file=sys.stderr)

    print(f"[TimesNet Subprocess] Training on device: {device}", file=sys.stderr)

    # Load data
    windows = np.array(data['windows'])
    labels = np.array(data['labels'])
    categories = data.get('categories')  # May be None

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    class_names = le.classes_.tolist()

    # Prepare data
    X = torch.FloatTensor(windows)
    if len(X.shape) == 2:
        X = X.unsqueeze(-1)
    y = torch.LongTensor(y_encoded)

    # Split data - prefer category-based split to avoid data leakage
    split_method = 'random'
    if categories is not None:
        categories = np.array(categories)
        train_mask = np.isin(categories, ['training', 'train'])
        test_mask = np.isin(categories, ['testing', 'test'])

        if np.sum(train_mask) > 0 and np.sum(test_mask) > 0:
            # Use category-based split (proper separation, no data leakage)
            split_method = 'category'
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            print(f"[TimesNet] Using CATEGORY-based split: {len(X_train)} train, {len(X_test)} test (no data leakage)", file=sys.stderr)
        else:
            print(f"[TimesNet] Categories provided but no proper train/test labels found. Falling back to random split.", file=sys.stderr)

    if split_method == 'random':
        # Fall back to random split (with small dataset handling)
        test_size = config.get('test_size', 0.2)
        n_samples = len(X)
        min_test_samples = num_classes  # Need at least 1 sample per class for stratification

        # Calculate actual test samples
        test_samples = int(n_samples * test_size) if test_size < 1 else int(test_size)

        # Adjust if dataset is too small for stratified split
        use_stratify = True
        if test_samples < min_test_samples:
            # Try to use minimum viable test_size
            adjusted_test_size = min_test_samples / n_samples
            if adjusted_test_size >= 0.5:
                # Dataset too small for stratified split, disable stratification
                print(f"[TimesNet] Dataset too small for stratified split ({n_samples} samples, {num_classes} classes). Using random split.", file=sys.stderr)
                use_stratify = False
            else:
                print(f"[TimesNet] Adjusted test_size from {test_size:.2f} to {adjusted_test_size:.2f} to ensure enough samples per class", file=sys.stderr)
                test_size = adjusted_test_size

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if use_stratify else None
        )
        print(f"[TimesNet] Using RANDOM split: {len(X_train)} train, {len(X_test)} test (WARNING: potential data leakage with overlapping windows)", file=sys.stderr)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Build model
    classifier_config = dict(config)
    classifier_config['num_classes'] = num_classes
    model = build_timesnet_classifier(classifier_config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    train_losses = []

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            output = model(batch_x)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{config['epochs']}] Loss: {avg_loss:.6f}", file=sys.stderr)

    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    y_true = y_test.numpy()

    # Import additional metrics
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Basic metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': class_names,
        'class_names': class_names,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'final_loss': float(train_losses[-1]) if train_losses else 0,
        'split_method': split_method,  # 'category' = proper split, 'random' = potential leakage
        'metrics_info': 'Category-based split (no data leakage)' if split_method == 'category' else 'Random split (potential data leakage with overlapping windows)'
    }

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['per_class_metrics'] = [
        {
            'class': class_names[i],
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
            'support': int(np.sum(y_true == i))
        }
        for i in range(num_classes)
    ]

    # ROC curve for binary classification or one-vs-rest for multiclass
    if num_classes == 2:
        # Binary classification - use probability of positive class
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        # Sample points for efficiency
        sample_indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
        metrics['roc_curve'] = {
            'fpr': [float(fpr[i]) for i in sample_indices],
            'tpr': [float(tpr[i]) for i in sample_indices],
            'auc': float(roc_auc)
        }
        metrics['roc_auc'] = float(roc_auc)

        # Precision-Recall curve
        prec, rec, _ = precision_recall_curve(y_true, y_probs[:, 1])
        pr_auc = average_precision_score(y_true, y_probs[:, 1])
        pr_sample_indices = np.linspace(0, len(prec) - 1, min(50, len(prec)), dtype=int)
        metrics['pr_curve'] = {
            'precision': [float(prec[i]) for i in pr_sample_indices],
            'recall': [float(rec[i]) for i in pr_sample_indices],
            'auc': float(pr_auc)
        }
        metrics['pr_auc'] = float(pr_auc)
    else:
        # Multiclass - use EXACT same approach as Traditional ML (ml_trainer.py lines 644-670)
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score

        # Binarize labels - use integer class list matching encoded labels
        class_indices = list(range(num_classes))
        y_true_bin = label_binarize(y_true, classes=class_indices)

        # Handle edge case where label_binarize returns (n, 1) for 2 classes
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        # Debug: Print shapes and value ranges
        print(f"[TimesNet DEBUG] y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}", file=sys.stderr)
        print(f"[TimesNet DEBUG] y_probs shape: {y_probs.shape}", file=sys.stderr)
        print(f"[TimesNet DEBUG] y_true_bin shape: {y_true_bin.shape}", file=sys.stderr)
        print(f"[TimesNet DEBUG] y_probs min/max: {y_probs.min():.4f}/{y_probs.max():.4f}", file=sys.stderr)
        print(f"[TimesNet DEBUG] y_probs row sums (should be ~1): {y_probs.sum(axis=1)[:5]}", file=sys.stderr)

        # Compute ROC curve per class and average (SAME as Traditional ML)
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(num_classes):
            if i < y_probs.shape[1] and i < y_true_bin.shape[1]:
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                class_auc_i = auc(fpr_i, tpr_i)
                print(f"[TimesNet DEBUG] Class {i} ({class_names[i]}): AUC={class_auc_i:.4f}, fpr_range=[{fpr_i.min():.3f},{fpr_i.max():.3f}], tpr_range=[{tpr_i.min():.3f},{tpr_i.max():.3f}]", file=sys.stderr)
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)

        mean_tpr /= num_classes
        print(f"[TimesNet DEBUG] mean_tpr range: [{mean_tpr.min():.4f}, {mean_tpr.max():.4f}]", file=sys.stderr)

        # Store the averaged curve
        metrics['roc_curve'] = {
            'fpr': all_fpr.tolist(),
            'tpr': mean_tpr.tolist()
        }

        # Compute AUC using sklearn (same as Traditional ML)
        try:
            computed_auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted'))
            metrics['auc_roc'] = computed_auc
            metrics['roc_auc'] = computed_auc
            print(f"[TimesNet] Multiclass ROC-AUC (sklearn ovr weighted): {computed_auc:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"[TimesNet] sklearn roc_auc_score failed: {e}", file=sys.stderr)
            # Fallback: compute from the averaged curve
            metrics['roc_auc'] = float(auc(all_fpr, mean_tpr))

    print(f"[TimesNet] Classification Metrics - Acc: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}, "
          f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}", file=sys.stderr)

    # Save model state
    model_state = {
        'model_state_dict': {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()},
        'config': config,
        'label_encoder_classes': class_names,
        'mode': 'classification'
    }

    return {
        'success': True,
        'metrics': metrics,
        'model_state': model_state,
        'device': str(device)
    }


def build_timesnet_encoder(config: dict):
    """Build TimesNet encoder for anomaly detection."""
    import torch.nn as nn

    class TimesNetEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.seq_len = config.get('seq_len', 128)
            self.enc_in = config.get('enc_in', 3)
            self.d_model = config.get('d_model', 64)
            self.d_ff = config.get('d_ff', 128)
            self.dropout = config.get('dropout', 0.1)

            # Embedding
            self.embed = nn.Linear(self.enc_in, self.d_model)

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_ff, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.d_ff, self.d_model, kernel_size=3, padding=1),
                nn.ReLU(),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_ff, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.d_ff, self.d_model, kernel_size=3, padding=1),
            )

            # Output projection
            self.projection = nn.Linear(self.d_model, self.enc_in)

        def forward(self, x):
            # x: [batch, seq_len, channels]
            x = self.embed(x)  # [batch, seq_len, d_model]
            x = x.transpose(1, 2)  # [batch, d_model, seq_len]
            x = self.encoder(x)
            x = self.decoder(x)
            x = x.transpose(1, 2)  # [batch, seq_len, d_model]
            x = self.projection(x)  # [batch, seq_len, channels]
            return x

    return TimesNetEncoder(config)


def build_timesnet_classifier(config: dict):
    """Build TimesNet classifier."""
    import torch.nn as nn

    class TimesNetClassifier(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.seq_len = config.get('seq_len', 128)
            self.enc_in = config.get('enc_in', 3)
            self.d_model = config.get('d_model', 64)
            self.d_ff = config.get('d_ff', 128)
            self.dropout = config.get('dropout', 0.1)
            self.num_classes = config.get('num_classes', 2)

            # Embedding
            self.embed = nn.Linear(self.enc_in, self.d_model)

            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_ff, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.d_ff, self.d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )

            # Global pooling and classifier
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.num_classes)
            )

        def forward(self, x):
            # x: [batch, seq_len, channels]
            x = self.embed(x)  # [batch, seq_len, d_model]
            x = x.transpose(1, 2)  # [batch, d_model, seq_len]
            x = self.encoder(x)  # [batch, d_model, seq_len]
            x = self.pool(x).squeeze(-1)  # [batch, d_model]
            x = self.classifier(x)  # [batch, num_classes]
            return x

    return TimesNetClassifier(config)


def check_gpu_status() -> dict:
    """Check GPU availability."""
    status = {
        'available': False,
        'cuda_available': False,
        'torch_available': False,
        'device_name': None,
        'error': None
    }

    try:
        import torch
        status['torch_available'] = True
        status['cuda_available'] = torch.cuda.is_available()

        if torch.cuda.is_available():
            status['available'] = True
            status['device_name'] = torch.cuda.get_device_name(0)
            status['device_count'] = torch.cuda.device_count()

            # Memory info
            props = torch.cuda.get_device_properties(0)
            status['memory_total_gb'] = round(props.total_memory / (1024**3), 2)

    except Exception as e:
        status['error'] = str(e)

    return status


def main():
    if len(sys.argv) < 3:
        print("Usage: python torch_subprocess.py <config_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # Load config
        with open(config_file, 'r') as f:
            job = json.load(f)

        task = job.get('task', 'train_anomaly')
        config = job.get('config', {})
        data = job.get('data', {})

        print(f"[TimesNet Subprocess] Starting task: {task}", file=sys.stderr)

        if task == 'check_gpu':
            result = check_gpu_status()
        elif task == 'train_anomaly':
            result = train_timesnet_anomaly(config, data)
        elif task == 'train_classification':
            result = train_timesnet_classification(config, data)
        else:
            result = {'success': False, 'error': f'Unknown task: {task}'}

        # Save result
        with open(output_file, 'w') as f:
            json.dump(result, f)

        print(f"[TimesNet Subprocess] Task completed successfully", file=sys.stderr)

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        with open(output_file, 'w') as f:
            json.dump(error_result, f)
        print(f"[TimesNet Subprocess] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
