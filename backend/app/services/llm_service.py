"""
CiRA ME - LLM Service
Handles local LLM inference using Ollama for feature recommendations
Supports GPU acceleration with CUDA
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for Ollama LLM service."""
    base_url: str = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    model: str = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')
    timeout: int = 120  # seconds
    temperature: float = 0.7
    max_tokens: int = 1024


class LLMService:
    """
    Service for interacting with local Ollama LLM.
    Supports Llama 3.2 with GPU acceleration.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._status_cache = None
        self._status_cache_time = 0

    def get_status(self) -> Dict[str, Any]:
        """
        Get Ollama service status and GPU information.

        Returns:
            Dictionary containing:
            - available: bool - whether Ollama is running
            - model: str - loaded model name
            - gpu_loaded: bool - whether model is loaded on GPU
            - gpu_info: dict - GPU details if available
        """
        import time

        # Cache status for 5 seconds
        if self._status_cache and (time.time() - self._status_cache_time) < 5:
            return self._status_cache

        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )

            if response.status_code != 200:
                return {
                    'available': False,
                    'model': None,
                    'gpu_loaded': False,
                    'error': 'Ollama service not responding'
                }

            models = response.json().get('models', [])
            # Get both full names and base names for matching
            model_names_full = [m.get('name', '') for m in models]
            model_names_base = [m.get('name', '').split(':')[0] for m in models]

            logger.info(f"Ollama models found: {model_names_full}")
            logger.info(f"Target model: {self.config.model}")

            # Check if our target model is available (match full name or base name)
            target_model_base = self.config.model.split(':')[0]
            model_available = (
                self.config.model in model_names_full or
                target_model_base in model_names_base
            )

            # Get running model info (GPU status)
            ps_response = requests.get(
                f"{self.config.base_url}/api/ps",
                timeout=5
            )

            gpu_loaded = False
            gpu_info = None

            if ps_response.status_code == 200:
                running_models = ps_response.json().get('models', [])
                logger.info(f"Running models from /api/ps: {running_models}")
                for model in running_models:
                    if target_model_base in model.get('name', ''):
                        # Check if model is using GPU
                        size_vram = model.get('size_vram', 0)
                        logger.info(f"Model {model.get('name')} size_vram: {size_vram}")
                        gpu_loaded = size_vram > 0
                        gpu_info = {
                            'vram_used_mb': size_vram / (1024 * 1024) if size_vram else 0,
                            'size_mb': model.get('size', 0) / (1024 * 1024),
                            'quantization': model.get('details', {}).get('quantization_level', 'unknown')
                        }
                        break

            status = {
                'available': True,
                'model': self.config.model,
                'model_installed': model_available,
                'gpu_loaded': gpu_loaded,
                'gpu_info': gpu_info,
                'available_models': model_names_full
            }

            self._status_cache = status
            self._status_cache_time = time.time()

            return status

        except requests.exceptions.ConnectionError:
            return {
                'available': False,
                'model': None,
                'gpu_loaded': False,
                'error': 'Cannot connect to Ollama. Is it running?'
            }
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return {
                'available': False,
                'model': None,
                'gpu_loaded': False,
                'error': str(e)
            }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context

        Returns:
            Dictionary with response text and metadata
        """
        try:
            messages = []

            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })

            messages.append({
                'role': 'user',
                'content': prompt
            })

            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    'model': self.config.model,
                    'messages': messages,
                    'stream': False,
                    'options': {
                        'temperature': self.config.temperature,
                        'num_predict': self.config.max_tokens
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f"Ollama API error: {response.status_code}",
                    'response': None
                }

            result = response.json()

            return {
                'success': True,
                'response': result.get('message', {}).get('content', ''),
                'model': result.get('model'),
                'eval_count': result.get('eval_count'),
                'eval_duration_ms': result.get('eval_duration', 0) / 1_000_000  # ns to ms
            }

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'LLM request timed out',
                'response': None
            }
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }

    def recommend_features(
        self,
        data_stats: Dict[str, Any],
        available_features: List[str],
        mode: str = 'anomaly',
        sensor_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get LLM-powered feature recommendations based on data characteristics.

        Args:
            data_stats: Statistics about the data (mean, std, range, etc.)
            available_features: List of available feature names
            mode: 'anomaly' or 'classification'
            sensor_info: Optional sensor metadata

        Returns:
            Dictionary with recommended features and reasoning
        """
        system_prompt = """You are an expert in signal processing and machine learning for time-series analysis.
Your task is to recommend the best features for the given task based on data characteristics.

Available feature categories:
- TSFresh statistical features: mean, std, min, max, median, sum, variance, skewness, kurtosis, abs_energy, root_mean_square, mean_abs_change, mean_change, count_above_mean, count_below_mean, first_location_of_maximum, first_location_of_minimum, last_location_of_maximum, last_location_of_minimum, percentage_of_reoccurring_values, sum_of_reoccurring_values, abs_sum_of_changes, range, interquartile_range, mean_second_derivative
- DSP features: rms, peak_to_peak, crest_factor, shape_factor, impulse_factor, margin_factor, zero_crossing_rate, autocorr_lag1, autocorr_lag5, binned_entropy, spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, spectral_entropy, peak_frequency, spectral_skewness, spectral_kurtosis, band_power_low, band_power_mid, band_power_high

Respond ONLY with valid JSON in this exact format:
{
  "recommended_features": ["feature1", "feature2", ...],
  "reasoning": ["reason1", "reason2", ...],
  "confidence": "high|medium|low"
}"""

        # Build the user prompt with data context
        data_summary = []
        if data_stats:
            if 'num_windows' in data_stats:
                data_summary.append(f"- Number of windows: {data_stats['num_windows']}")
            if 'window_size' in data_stats:
                data_summary.append(f"- Window size: {data_stats['window_size']} samples")
            if 'num_channels' in data_stats:
                data_summary.append(f"- Number of sensor channels: {data_stats['num_channels']}")
            if 'label_distribution' in data_stats:
                data_summary.append(f"- Label distribution: {data_stats['label_distribution']}")
            if 'sampling_rate' in data_stats:
                data_summary.append(f"- Sampling rate: {data_stats['sampling_rate']} Hz")

        sensor_desc = ""
        if sensor_info:
            sensor_desc = f"\nSensor information: {sensor_info.get('description', 'Accelerometer/vibration data')}"

        user_prompt = f"""Task: {mode.upper()} detection/classification

Data characteristics:
{chr(10).join(data_summary) if data_summary else '- Standard time-series sensor data'}
{sensor_desc}

Please recommend 8-12 features from the available list that would be most effective for this {mode} task.
Consider:
1. Features that capture signal distribution (for anomaly detection)
2. Features that capture frequency content (for classification)
3. Features that are computationally efficient
4. Avoid redundant features

Respond with JSON only."""

        # Try LLM generation
        result = self.generate(user_prompt, system_prompt)

        if not result['success']:
            # Fallback to rule-based recommendations
            return self._fallback_recommendations(mode)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = result['response']
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            recommendations = json.loads(response_text.strip())

            # Validate recommended features exist
            valid_features = [f for f in recommendations.get('recommended_features', [])
                           if f in available_features]

            if not valid_features:
                return self._fallback_recommendations(mode)

            return {
                'recommended_features': valid_features,
                'reasoning': recommendations.get('reasoning', []),
                'confidence': recommendations.get('confidence', 'medium'),
                'mode': mode,
                'llm_used': True,
                'model': self.config.model,
                'total_recommended': len(valid_features)
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return self._fallback_recommendations(mode)

    def _fallback_recommendations(self, mode: str) -> Dict[str, Any]:
        """Fallback rule-based recommendations when LLM is unavailable."""
        recommendations = ['mean', 'std', 'rms', 'min', 'max']
        reasoning = ["Basic statistical features provide fundamental signal characteristics."]

        if mode == 'anomaly':
            recommendations.extend([
                'kurtosis', 'skewness', 'peak_to_peak', 'crest_factor',
                'spectral_entropy', 'zero_crossing_rate', 'abs_energy'
            ])
            reasoning.append("Distribution features (kurtosis, skewness) help detect unusual patterns.")
            reasoning.append("Energy features identify abnormal signal power levels.")
        else:
            recommendations.extend([
                'spectral_centroid', 'spectral_bandwidth', 'peak_frequency',
                'band_power_low', 'band_power_mid', 'band_power_high',
                'shape_factor', 'impulse_factor'
            ])
            reasoning.append("Frequency features help distinguish different signal classes.")
            reasoning.append("Shape factors capture waveform characteristics unique to each class.")

        return {
            'recommended_features': list(set(recommendations)),
            'reasoning': reasoning,
            'mode': mode,
            'llm_used': False,
            'confidence': 'medium',
            'total_recommended': len(set(recommendations))
        }


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the singleton LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
