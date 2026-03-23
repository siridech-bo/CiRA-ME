"""
CiRA ME - Custom Model Runner Service
Executes user-submitted Python model code in an isolated subprocess.
"""

import os
import sys
import json
import uuid
import pickle
import tempfile
import subprocess
import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from .feature_extractor import FeatureExtractor, _feature_sessions
from .ml_trainer import _model_sessions

logger = logging.getLogger(__name__)

# Base class code injected into the sandbox
CIRA_BASE_CLASS = '''
import numpy as np

class CiraModel:
    """Base class for custom CiRA ME models.

    Subclass this and implement the required methods.
    CiRA fills in the attributes before calling build().
    """

    task = None          # "classification" | "anomaly" | "regression"
    n_features = 0       # number of input features
    n_classes = 0        # number of classes (0 for regression)
    class_names = []     # class label names

    def build(self, config: dict):
        """Create your model architecture. Called once before training."""
        raise NotImplementedError("Implement build() to create your model")

    def train(self, X_train, y_train, X_val, y_val) -> dict:
        """Train your model and return a metrics dict.

        Args:
            X_train: Training features (numpy array)
            y_train: Training labels/targets (numpy array)
            X_val: Validation features (numpy array)
            y_val: Validation labels/targets (numpy array)

        Returns:
            dict with metric names -> values (e.g. {"accuracy": 0.95, "f1": 0.93})
        """
        raise NotImplementedError("Implement train() to train your model")

    def predict(self, X) -> np.ndarray:
        """Return predictions for input features.

        Args:
            X: Input features (numpy array)

        Returns:
            numpy array of predictions
        """
        raise NotImplementedError("Implement predict() to make predictions")

    def get_model(self):
        """Return the underlying model object for export (ONNX/pickle).

        Returns:
            The trained model object (sklearn estimator, torch.nn.Module, etc.)
        """
        raise NotImplementedError("Implement get_model() to return the model for export")
'''

# Runner script executed in subprocess
RUNNER_SCRIPT = '''
import sys
import os
import json
import pickle
import traceback
import numpy as np

# Load config
config_path = sys.argv[1]
output_path = sys.argv[2]

with open(config_path, 'r') as f:
    config = json.load(f)

data_path = config['data_path']
code = config['code']
task = config['task']
n_features = config['n_features']
n_classes = config['n_classes']
class_names = config['class_names']
user_config = config.get('user_config', {})

result = {'status': 'error', 'error': None, 'metrics': {}, 'logs': []}

try:
    # Load data
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    result['logs'].append(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {n_features} features")

    # Execute user code (defines their model class)
    exec_globals = {'__builtins__': __builtins__, 'np': np, 'numpy': np}

    # Inject CiraModel base class
    base_code = open(os.path.join(os.path.dirname(config_path), 'cira_base.py'), 'r').read()
    exec(base_code, exec_globals)

    # Execute user code
    exec(code, exec_globals)

    # Find the user's model class (subclass of CiraModel)
    CiraModel = exec_globals['CiraModel']
    user_class = None
    for name, obj in exec_globals.items():
        if isinstance(obj, type) and issubclass(obj, CiraModel) and obj is not CiraModel:
            user_class = obj
            break

    if user_class is None:
        raise ValueError("No CiraModel subclass found in your code. Define a class that inherits from CiraModel.")

    result['logs'].append(f"Found model class: {user_class.__name__}")

    # Instantiate and configure
    model = user_class()
    model.task = task
    model.n_features = n_features
    model.n_classes = n_classes
    model.class_names = class_names

    # Build
    result['logs'].append("Building model...")
    model.build(user_config)

    # Train
    result['logs'].append("Training...")
    metrics = model.train(X_train, y_train, X_val, y_val)
    if not isinstance(metrics, dict):
        metrics = {'custom_metric': float(metrics) if metrics is not None else 0}

    result['metrics'] = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                        for k, v in metrics.items()}
    result['logs'].append(f"Training complete. Metrics: {result['metrics']}")

    # Save the model object
    model_obj = model.get_model()
    model_output_path = os.path.join(os.path.dirname(output_path), 'model.pkl')
    with open(model_output_path, 'wb') as f:
        pickle.dump(model_obj, f)
    result['model_path'] = model_output_path

    # Try to get predictions for validation metrics
    try:
        y_pred = model.predict(X_val)
        result['predictions_shape'] = list(y_pred.shape)
    except Exception as pred_err:
        result['logs'].append(f"Warning: predict() failed: {pred_err}")

    result['status'] = 'success'
    result['logs'].append("Done.")

except Exception as e:
    result['status'] = 'error'
    result['error'] = str(e)
    result['traceback'] = traceback.format_exc()
    result['logs'].append(f"Error: {e}")

# Write output
with open(output_path, 'w') as f:
    json.dump(result, f)
'''


class CustomModelRunner:
    """Runs user-submitted model code in an isolated subprocess."""

    def __init__(self, models_path: str = './models'):
        self.models_path = models_path
        os.makedirs(models_path, exist_ok=True)

    def execute(
        self,
        code: str,
        feature_session_id: str,
        task: str = 'classification',
        test_size: float = 0.2,
        user_config: Dict = None,
        timeout: int = 300,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute user's custom model code.

        Args:
            code: User's Python code defining a CiraModel subclass
            feature_session_id: Session with extracted features
            task: 'classification', 'anomaly', or 'regression'
            test_size: Validation split ratio
            user_config: Optional config dict passed to model.build()
            timeout: Max execution time in seconds
            user_id: User ID for tracking

        Returns:
            Dict with status, metrics, logs, and training_session_id
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        user_config = user_config or {}

        # Get feature data
        X, y, categories = FeatureExtractor.get_features_for_training(feature_session_id)

        if y is None and task != 'anomaly':
            raise ValueError("Labels required for classification/regression")

        # Split data
        if categories is not None:
            train_mask = np.isin(categories, ['training', 'train'])
            test_mask = np.isin(categories, ['testing', 'test'])
            if np.sum(train_mask) > 0 and np.sum(test_mask) > 0:
                X_train, X_val = X[train_mask], X[test_mask]
                y_train = y[train_mask] if y is not None else np.zeros(np.sum(train_mask))
                y_val = y[test_mask] if y is not None else np.zeros(np.sum(test_mask))
            else:
                if y is not None:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                else:
                    X_train, X_val = train_test_split(X, test_size=test_size, random_state=42)
                    y_train = np.zeros(len(X_train))
                    y_val = np.zeros(len(X_val))
        else:
            if y is not None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            else:
                X_train, X_val = train_test_split(X, test_size=test_size, random_state=42)
                y_train = np.zeros(len(X_train))
                y_val = np.zeros(len(X_val))

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Determine class info
        if y is not None:
            classes = np.unique(y)
            n_classes = len(classes)
            class_names = [str(c) for c in classes]
        else:
            n_classes = 0
            class_names = []

        # Prepare temp directory
        tmp_dir = tempfile.mkdtemp(prefix='cira_custom_')
        data_path = os.path.join(tmp_dir, 'data.npz')
        config_path = os.path.join(tmp_dir, 'config.json')
        output_path = os.path.join(tmp_dir, 'output.json')
        runner_path = os.path.join(tmp_dir, 'runner.py')
        base_path = os.path.join(tmp_dir, 'cira_base.py')

        try:
            # Save data
            np.savez(data_path,
                     X_train=X_train_scaled,
                     y_train=y_train,
                     X_val=X_val_scaled,
                     y_val=y_val)

            # Save config
            config = {
                'data_path': data_path,
                'code': code,
                'task': task,
                'n_features': int(X_train.shape[1]),
                'n_classes': int(n_classes),
                'class_names': class_names,
                'user_config': user_config,
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Save base class and runner
            with open(base_path, 'w') as f:
                f.write(CIRA_BASE_CLASS)
            with open(runner_path, 'w') as f:
                f.write(RUNNER_SCRIPT)

            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, runner_path, config_path, output_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir
            )

            # Read output
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    run_result = json.load(f)
            else:
                run_result = {
                    'status': 'error',
                    'error': result.stderr or 'No output produced',
                    'logs': [result.stdout] if result.stdout else [],
                }

            # If successful, store model in session
            if run_result.get('status') == 'success' and run_result.get('model_path'):
                training_session_id = str(uuid.uuid4())

                # Move model to permanent location
                perm_model_path = os.path.join(self.models_path, f"{training_session_id}.pkl")
                model_pkl_path = run_result['model_path']

                if os.path.exists(model_pkl_path):
                    with open(model_pkl_path, 'rb') as f:
                        model_obj = pickle.load(f)

                    # Save with scaler
                    with open(perm_model_path, 'wb') as f:
                        pickle.dump({
                            'model': model_obj,
                            'scaler': scaler,
                            'algorithm': 'custom',
                            'mode': task,
                            'hyperparameters': user_config,
                            'feature_session_id': feature_session_id,
                            'feature_names': _feature_sessions.get(feature_session_id, {}).get('feature_names', []),
                        }, f)

                    _model_sessions[training_session_id] = {
                        'model': model_obj,
                        'scaler': scaler,
                        'algorithm': 'custom',
                        'mode': task,
                        'metrics': run_result.get('metrics', {}),
                        'model_path': perm_model_path,
                        'hyperparameters': user_config,
                        'created_at': datetime.utcnow().isoformat()
                    }

                    run_result['training_session_id'] = training_session_id
                    run_result['model_path'] = perm_model_path

            # Add stdout/stderr
            if result.stdout:
                run_result.setdefault('logs', [])
                run_result['logs'].extend(result.stdout.strip().split('\n'))
            if result.stderr:
                run_result.setdefault('logs', [])
                run_result['logs'].extend(['[stderr] ' + line for line in result.stderr.strip().split('\n')])

            return run_result

        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'error': f'Execution timed out after {timeout} seconds',
                'logs': ['Execution killed due to timeout'],
            }
        except Exception as e:
            logger.error(f"Custom model execution error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'logs': [],
            }
        finally:
            # Cleanup temp files (but not the permanent model)
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
