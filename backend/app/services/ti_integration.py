"""
CiRA ME - TI TinyML ModelMaker Integration Service
Bridges CiRA ME backend to the TI ModelMaker container.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# TI ModelMaker service URL (Docker internal network)
TI_SERVICE_URL = os.environ.get('TI_MODELMAKER_URL', 'http://cirame-ti-modelmaker:5200')


class TIIntegration:
    """Client for TI TinyML ModelMaker service."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or TI_SERVICE_URL
        self.timeout = 30  # seconds for API calls (not training)

    def is_available(self) -> bool:
        """Check if TI ModelMaker service is running."""
        try:
            resp = requests.get(f'{self.base_url}/health', timeout=5)
            data = resp.json()
            return data.get('status') in ('healthy', 'degraded')
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        """Get health status of TI service."""
        try:
            resp = requests.get(f'{self.base_url}/health', timeout=5)
            return resp.json()
        except Exception as e:
            return {'status': 'offline', 'error': str(e)}

    def get_devices(self) -> Dict[str, Any]:
        """Get supported TI MCU devices."""
        resp = requests.get(f'{self.base_url}/devices', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_models(self, task: str, device: str = None,
                   source: str = 'all') -> Dict[str, Any]:
        """Get available models from TI model zoo."""
        params = {'task': task, 'source': source}
        if device:
            params['device'] = device
        resp = requests.get(f'{self.base_url}/models', params=params,
                            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def train(self, task_type: str, target_device: str,
              dataset_path: str, config: Dict = None,
              model_names: list = None, model_name: str = None) -> Dict[str, Any]:
        """Start training via TI ModelMaker.

        Args:
            task_type: 'timeseries_regression', 'timeseries_classification', etc.
            model_names: List of models to train
            target_device: TI device ID (e.g., 'F2837')
            dataset_path: Path to CSV dataset (must be accessible in TI container)
            config: Optional training overrides (epochs, batch_size, etc.)

        Returns:
            Training result with metrics, logs, and artifact paths
        """
        payload = {
            'task_type': task_type,
            'model_names': model_names or ([model_name] if model_name else []),
            'target_device': target_device,
            'dataset_path': dataset_path,
            'config': config or {},
        }

        # Training can take a long time
        resp = requests.post(f'{self.base_url}/train', json=payload, timeout=660)
        resp.raise_for_status()
        return resp.json()

    def download_artifacts(self, run_id: str) -> bytes:
        """Download compiled model artifacts as zip bytes."""
        resp = requests.get(f'{self.base_url}/download/{run_id}',
                            timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def map_cira_mode_to_ti_task(mode: str) -> str:
        """Map CiRA ME mode to TI task type."""
        mapping = {
            'anomaly': 'timeseries_anomalydetection',
            'classification': 'timeseries_classification',
            'regression': 'timeseries_regression',
        }
        return mapping.get(mode, 'timeseries_regression')
