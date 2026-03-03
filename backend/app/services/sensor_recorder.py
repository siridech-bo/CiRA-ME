"""
CiRA ME - System Sensor Recorder
Records CPU, RAM, disk, network, and GPU sensors as time series CSV
for testing the ML pipeline end-to-end.
"""

import os
import csv
import time
import uuid
import threading
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

# Global storage for recording jobs
_recording_jobs: Dict[str, Dict] = {}


class StressGenerator:
    """Generate CPU/IO/GPU stress to create distinct sensor patterns."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    def cpu_stress(self, cores: Optional[int] = None):
        """Spin busy-loop threads on all cores."""
        cores = cores or os.cpu_count() or 4
        for _ in range(cores):
            t = threading.Thread(target=self._cpu_worker, daemon=True)
            t.start()
            self._threads.append(t)

    def io_stress(self):
        """Write/read temp file blocks to spike disk I/O."""
        for _ in range(2):
            t = threading.Thread(target=self._io_worker, daemon=True)
            t.start()
            self._threads.append(t)

    def gpu_stress(self):
        """Run matrix multiplications on CUDA GPU."""
        t = threading.Thread(target=self._gpu_worker, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        """Signal all stress threads to stop."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()
        self._stop_event.clear()

    def _cpu_worker(self):
        while not self._stop_event.is_set():
            sum(i * i for i in range(10000))

    def _io_worker(self):
        block = b'\x00' * (1024 * 1024)  # 1 MB
        while not self._stop_event.is_set():
            try:
                with tempfile.NamedTemporaryFile(delete=True) as f:
                    for _ in range(10):
                        if self._stop_event.is_set():
                            break
                        f.write(block)
                    f.flush()
                    f.seek(0)
                    while f.read(1024 * 1024):
                        if self._stop_event.is_set():
                            break
            except Exception:
                break

    def _gpu_worker(self):
        try:
            import torch
            if not torch.cuda.is_available():
                return
            device = torch.device('cuda')
            a = torch.randn(2048, 2048, device=device)
            while not self._stop_event.is_set():
                torch.mm(a, a)
                torch.cuda.synchronize()
        except Exception:
            return


class SensorRecorder:
    """Record system sensor data as time series CSV."""

    def __init__(self):
        if psutil is None:
            raise ImportError("psutil is required. Install with: pip install psutil")
        self._gpu_method: Optional[str] = None
        self._gpu_handle = None
        self._prev_disk = None
        self._prev_net = None
        self._prev_time = None
        self._init_gpu()

    def _init_gpu(self):
        """Try to initialise GPU monitoring."""
        # Try pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_method = 'pynvml'
            return
        except Exception:
            pass

        # Try nvidia-smi CLI
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                self._gpu_method = 'nvidia-smi'
                return
        except Exception:
            pass

        self._gpu_method = None

    def _prime_counters(self):
        """Initialise baselines for cpu_percent and I/O deltas."""
        psutil.cpu_percent(interval=None)
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
        self._prev_time = time.perf_counter()
        time.sleep(0.5)

    def _read_sensors(self) -> Dict[str, float]:
        """Take a single sensor reading."""
        now = time.perf_counter()
        dt = now - self._prev_time if self._prev_time else 1.0
        if dt <= 0:
            dt = 1.0

        row: Dict[str, float] = {}

        # CPU
        row['cpu_percent'] = psutil.cpu_percent(interval=None)
        freq = psutil.cpu_freq()
        row['cpu_freq_mhz'] = round(freq.current, 1) if freq else 0.0

        # RAM
        mem = psutil.virtual_memory()
        row['ram_percent'] = mem.percent
        row['ram_used_gb'] = round(mem.used / (1024 ** 3), 2)

        # Disk I/O (delta rate)
        disk = psutil.disk_io_counters()
        if self._prev_disk and disk:
            row['disk_read_mb_s'] = round(
                (disk.read_bytes - self._prev_disk.read_bytes) / dt / (1024 * 1024), 3
            )
            row['disk_write_mb_s'] = round(
                (disk.write_bytes - self._prev_disk.write_bytes) / dt / (1024 * 1024), 3
            )
        else:
            row['disk_read_mb_s'] = 0.0
            row['disk_write_mb_s'] = 0.0
        self._prev_disk = disk

        # Network I/O (delta rate)
        net = psutil.net_io_counters()
        if self._prev_net and net:
            row['net_sent_mb_s'] = round(
                (net.bytes_sent - self._prev_net.bytes_sent) / dt / (1024 * 1024), 3
            )
            row['net_recv_mb_s'] = round(
                (net.bytes_recv - self._prev_net.bytes_recv) / dt / (1024 * 1024), 3
            )
        else:
            row['net_sent_mb_s'] = 0.0
            row['net_recv_mb_s'] = 0.0
        self._prev_net = net

        # GPU
        if self._gpu_method == 'pynvml':
            row.update(self._read_gpu_pynvml())
        elif self._gpu_method == 'nvidia-smi':
            row.update(self._read_gpu_smi())

        self._prev_time = now
        return row

    def _read_gpu_pynvml(self) -> Dict[str, float]:
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            mem_pct = round(mem_info.used / mem_info.total * 100, 1) if mem_info.total > 0 else 0.0
            return {
                'gpu_util_percent': float(util.gpu),
                'gpu_mem_percent': mem_pct,
                'gpu_temp_c': float(temp),
                'gpu_power_w': round(power, 1),
            }
        except Exception:
            return {
                'gpu_util_percent': 0.0,
                'gpu_mem_percent': 0.0,
                'gpu_temp_c': 0.0,
                'gpu_power_w': 0.0,
            }

    def _read_gpu_smi(self) -> Dict[str, float]:
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(',')]
                return {
                    'gpu_util_percent': float(parts[0]),
                    'gpu_mem_percent': float(parts[1]),
                    'gpu_temp_c': float(parts[2]),
                    'gpu_power_w': float(parts[3]),
                }
        except Exception:
            pass
        return {
            'gpu_util_percent': 0.0,
            'gpu_mem_percent': 0.0,
            'gpu_temp_c': 0.0,
            'gpu_power_w': 0.0,
        }

    @property
    def has_gpu(self) -> bool:
        return self._gpu_method is not None

    @property
    def sensor_columns(self) -> List[str]:
        cols = [
            'cpu_percent', 'cpu_freq_mhz', 'ram_percent', 'ram_used_gb',
            'disk_read_mb_s', 'disk_write_mb_s', 'net_sent_mb_s', 'net_recv_mb_s',
        ]
        if self.has_gpu:
            cols += ['gpu_util_percent', 'gpu_mem_percent', 'gpu_temp_c', 'gpu_power_w']
        return cols

    def record(self, job_id: str, duration: float, rate: float,
               label_schedule: List[Dict], output_path: str):
        """Record sensors to CSV. Runs in a thread.

        label_schedule: list of {"start_pct": 0.0, "end_pct": 0.4, "label": "Normal",
                                  "stress": None|"cpu"|"io"|"gpu"}
        """
        job = _recording_jobs[job_id]
        total_samples = int(duration * rate)
        sample_interval = 1.0 / rate

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # CSV columns
        header = ['time'] + self.sensor_columns + ['label']

        self._prime_counters()

        stress = StressGenerator()
        rows = []
        current_phase_idx = -1
        start_time = time.perf_counter()

        try:
            for i in range(total_samples):
                if job.get('stop_requested'):
                    break

                elapsed_pct = i / total_samples if total_samples > 0 else 0.0
                sample_time = round(i * sample_interval, 6)

                # Determine current phase
                phase_idx = 0
                current_label = label_schedule[0]['label']
                for idx, phase in enumerate(label_schedule):
                    if phase['start_pct'] <= elapsed_pct < phase['end_pct']:
                        phase_idx = idx
                        current_label = phase['label']
                        break

                # Start/stop stress when phase changes
                if phase_idx != current_phase_idx:
                    stress.stop()
                    current_phase_idx = phase_idx
                    phase = label_schedule[phase_idx]
                    job['current_phase'] = phase['label']

                    stress_type = phase.get('stress')
                    if stress_type == 'cpu':
                        stress.cpu_stress()
                    elif stress_type == 'io':
                        stress.io_stress()
                    elif stress_type == 'gpu' and self.has_gpu:
                        stress.gpu_stress()

                # Read sensors
                sensors = self._read_sensors()
                row = {'time': sample_time, 'label': current_label}
                row.update(sensors)
                rows.append(row)

                # Update progress
                job['elapsed'] = sample_time
                job['samples_collected'] = i + 1

                # Sleep until next sample
                target = start_time + (i + 1) * sample_interval
                sleep_time = target - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            stress.stop()

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                # Only write columns in header
                writer.writerow({k: row.get(k, '') for k in header})

        job['status'] = 'completed'
        job['output_path'] = output_path
        job['total_samples'] = len(rows)
        job['completed_at'] = datetime.utcnow().isoformat()


def build_label_schedule(mode: str, label: str = 'Normal',
                         has_gpu: bool = False) -> List[Dict]:
    """Build a label schedule based on recording mode."""
    if mode == 'manual':
        return [{'start_pct': 0.0, 'end_pct': 1.0, 'label': label, 'stress': None}]

    elif mode == 'anomaly':
        return [
            {'start_pct': 0.0, 'end_pct': 0.4, 'label': 'Normal', 'stress': None},
            {'start_pct': 0.4, 'end_pct': 0.8, 'label': 'Anomaly', 'stress': 'cpu'},
            {'start_pct': 0.8, 'end_pct': 1.0, 'label': 'Normal', 'stress': None},
        ]

    elif mode == 'classify':
        if has_gpu:
            return [
                {'start_pct': 0.0, 'end_pct': 0.25, 'label': 'idle', 'stress': None},
                {'start_pct': 0.25, 'end_pct': 0.5, 'label': 'cpu_stress', 'stress': 'cpu'},
                {'start_pct': 0.5, 'end_pct': 0.75, 'label': 'io_stress', 'stress': 'io'},
                {'start_pct': 0.75, 'end_pct': 1.0, 'label': 'gpu_stress', 'stress': 'gpu'},
            ]
        else:
            return [
                {'start_pct': 0.0, 'end_pct': 0.333, 'label': 'idle', 'stress': None},
                {'start_pct': 0.333, 'end_pct': 0.667, 'label': 'cpu_stress', 'stress': 'cpu'},
                {'start_pct': 0.667, 'end_pct': 1.0, 'label': 'io_stress', 'stress': 'io'},
            ]

    raise ValueError(f"Unknown mode: {mode}")


def start_recording(mode: str, duration: float, rate: float,
                    label: str, output_dir: str,
                    filename: Optional[str] = None) -> Dict[str, Any]:
    """Start a sensor recording in a background thread. Returns job info."""
    job_id = str(uuid.uuid4())
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    if filename:
        fname = f"{filename}.csv"
    else:
        fname = f"sensor_{mode}_{ts}.csv"

    output_path = os.path.join(output_dir, fname)

    recorder = SensorRecorder()
    schedule = build_label_schedule(mode, label, has_gpu=recorder.has_gpu)

    job = {
        'id': job_id,
        'status': 'recording',
        'mode': mode,
        'duration': duration,
        'rate': rate,
        'elapsed': 0.0,
        'samples_collected': 0,
        'total_expected': int(duration * rate),
        'current_phase': schedule[0]['label'],
        'phases': [p['label'] for p in schedule],
        'has_gpu': recorder.has_gpu,
        'output_path': output_path,
        'output_filename': fname,
        'started_at': datetime.utcnow().isoformat(),
        'stop_requested': False,
    }
    _recording_jobs[job_id] = job

    thread = threading.Thread(
        target=recorder.record,
        args=(job_id, duration, rate, schedule, output_path),
        daemon=True,
    )
    thread.start()

    return job


def get_recording_status(job_id: str) -> Optional[Dict]:
    return _recording_jobs.get(job_id)


def stop_recording(job_id: str) -> Optional[Dict]:
    job = _recording_jobs.get(job_id)
    if job and job['status'] == 'recording':
        job['stop_requested'] = True
    return job
