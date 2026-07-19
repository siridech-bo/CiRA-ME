"""CiRA ME - Sensor / hierarchy preset data (Phase A, 2026-07-18).

Static dropdown values served by GET /api/asset-tree/presets. Keep this
file in sync with the frontend's preset picker.
"""

UNIT_PRESETS = [
    {'value': 'celsius', 'label': '°C'},
    {'value': 'fahrenheit', 'label': '°F'},
    {'value': 'kelvin', 'label': 'K'},
    {'value': 'mm_s2', 'label': 'mm/s² (vibration)'},
    {'value': 'g', 'label': 'g (accel)'},
    {'value': 'hz', 'label': 'Hz'},
    {'value': 'rpm', 'label': 'RPM'},
    {'value': 'percent', 'label': '%'},
    {'value': 'pa', 'label': 'Pa (pressure)'},
    {'value': 'kpa', 'label': 'kPa'},
    {'value': 'bar', 'label': 'bar'},
    {'value': 'v', 'label': 'V'},
    {'value': 'a', 'label': 'A'},
    {'value': 'w', 'label': 'W'},
    {'value': 'custom', 'label': 'Custom…'},
]

SAMPLE_RATE_PRESETS = [10, 50, 100, 250, 500, 1000, 2000]

SENSOR_TEMPLATES = [
    {'value': 'vibration_monitor', 'label': 'Vibration monitor',
     'sensors': [
       {'name': 'accel_x', 'unit': 'mm_s2', 'sample_rate_hz': 1000, 'data_type': 'float'},
       {'name': 'accel_y', 'unit': 'mm_s2', 'sample_rate_hz': 1000, 'data_type': 'float'},
       {'name': 'accel_z', 'unit': 'mm_s2', 'sample_rate_hz': 1000, 'data_type': 'float'},
     ]},
    {'value': 'thermal', 'label': 'Thermal',
     'sensors': [
       {'name': 'temperature', 'unit': 'celsius', 'sample_rate_hz': 10, 'data_type': 'float'},
       {'name': 'humidity',    'unit': 'percent', 'sample_rate_hz': 10, 'data_type': 'float'},
     ]},
    {'value': 'rotating_machinery', 'label': 'Rotating machinery',
     'sensors': [
       {'name': 'rpm',         'unit': 'rpm',       'sample_rate_hz': 100, 'data_type': 'float'},
       {'name': 'vibration',   'unit': 'mm_s2',     'sample_rate_hz': 1000, 'data_type': 'float'},
       {'name': 'temperature', 'unit': 'celsius',   'sample_rate_hz': 10,  'data_type': 'float'},
       {'name': 'current',     'unit': 'a',         'sample_rate_hz': 100, 'data_type': 'float'},
     ]},
    {'value': 'blank', 'label': 'Blank', 'sensors': []},
]

HIERARCHY_PRESETS = [
    {'value': 'factory', 'label': 'Factory', 'levels': ['factory', 'plant', 'machine', 'sensor']},
    {'value': 'hospital', 'label': 'Hospital', 'levels': ['hospital', 'ward', 'bed', 'device']},
    {'value': 'fleet', 'label': 'Fleet', 'levels': ['fleet', 'vehicle', 'subsystem', 'channel']},
    {'value': 'farm', 'label': 'Farm', 'levels': ['farm', 'field', 'greenhouse', 'probe']},
    {'value': 'custom', 'label': 'Custom', 'levels': []},
]
