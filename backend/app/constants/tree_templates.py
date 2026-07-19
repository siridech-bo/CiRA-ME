"""
Ready-to-use starter trees. Users pick one in the wizard Step 1 (or from
the admin view when the tree is empty) and get 90% of their asset tree
built in a single click. Each template ships with its own level_names,
root_name, and a full nested tree including sensor metadata on the leaves.

Grouped into 4 categories + a Blank starter.

Adding a template:
- Bump the list below with a full dict following the shape used by the
  existing entries. `id` must be a stable slug (used in the apply URL).
- Sensor metadata `unit` values must match one of `UNIT_PRESETS` in
  sensor_presets.py, otherwise the wizard's dropdown will show "Custom..."
  as the effective value.
- Keep templates realistic-but-small — ~10-40 leaves is the sweet spot.
  Bigger trees just get more painful to rename, and users can duplicate
  branches later.

Selecting a template in the wizard skips Steps 2 (level names) and 3
(tree builder) because both are already defined by the template. Only
Step 4 (MQTT rules) requires per-install input.
"""

# ── Helpers ────────────────────────────────────────────────────────────────
# These reduce boilerplate across ~1400 lines of template definitions.
# Each factory returns a plain dict matching the shape expected by
# POST /api/asset-tree/import — same schema used by the wizard.


def _sensor(name, unit, sample_rate=100, data_type='float'):
    """Leaf node with sensor_meta populated."""
    return {
        'name': name,
        'sensor_meta': {
            'unit': unit,
            'sample_rate_hz': sample_rate,
            'data_type': data_type,
        },
    }


def _node(name, children=None, display_name=None, description=None):
    """Non-leaf node."""
    d = {'name': name}
    if display_name:
        d['display_name'] = display_name
    if description:
        d['description'] = description
    if children:
        d['children'] = children
    return d


DEFAULT_META = ['_meta', '_health', '_config', '_cmd']


# ── Industrial IoT ─────────────────────────────────────────────────────────

def _small_factory():
    def machine(i):
        return _node(f'machine_{i}', children=[
            _sensor('temperature', 'celsius', 10),
            _sensor('vibration', 'mm_s2', 1000),
            _sensor('pressure', 'kpa', 10),
        ])
    return {
        'id': 'small_factory',
        'name': 'Small factory',
        'category': 'industrial',
        'category_label': 'Industrial IoT',
        'icon': 'mdi-factory',
        'description': '1 plant, 3 machines. Each machine reports temperature, vibration, and pressure — a solid baseline for a first anomaly / classification model.',
        'config': {
            'level_names': ['factory', 'plant', 'machine', 'sensor'],
            'root_name': 'factory',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('factory', children=[
            _node('plant_A', children=[machine(i) for i in range(1, 4)]),
        ]),
    }


def _vibration_rig():
    def machine(i):
        return _node(f'machine_{i}', children=[
            _sensor('accel_x', 'mm_s2', 1000),
            _sensor('accel_y', 'mm_s2', 1000),
            _sensor('accel_z', 'mm_s2', 1000),
        ])
    return {
        'id': 'vibration_rig',
        'name': 'Vibration monitoring rig',
        'category': 'industrial',
        'category_label': 'Industrial IoT',
        'icon': 'mdi-vibrate',
        'description': '1 plant, 5 machines. Each machine has a 3-axis accelerometer at 1 kHz — ideal for bearing-fault classifiers and imbalance detection.',
        'config': {
            'level_names': ['factory', 'plant', 'machine', 'sensor'],
            'root_name': 'factory',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('factory', children=[
            _node('plant_A', children=[machine(i) for i in range(1, 6)]),
        ]),
    }


def _rotating_fleet():
    def machine(i):
        return _node(f'machine_{i}', children=[
            _sensor('rpm', 'rpm', 100),
            _sensor('vibration', 'mm_s2', 1000),
            _sensor('temperature', 'celsius', 10),
            _sensor('current', 'a', 100),
        ])
    return {
        'id': 'rotating_fleet',
        'name': 'Rotating machinery fleet',
        'category': 'industrial',
        'category_label': 'Industrial IoT',
        'icon': 'mdi-fan',
        'description': '1 plant, 3 rotating machines with RPM, vibration, temperature, and current draw. Good for predictive maintenance stories.',
        'config': {
            'level_names': ['factory', 'plant', 'machine', 'sensor'],
            'root_name': 'factory',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('factory', children=[
            _node('plant_A', children=[machine(i) for i in range(1, 4)]),
        ]),
    }


# ── Business Operations ────────────────────────────────────────────────────

def _retail_stores():
    def checkout(i):
        return _node(f'checkout_{i}', children=[
            _sensor('foot_traffic', 'custom', 1, 'int'),  # people per interval
            _sensor('transactions', 'custom', 1, 'int'),
            _sensor('queue_time', 'custom', 1, 'float'),  # seconds
        ])
    def store(i):
        return _node(f'store_{i}', children=[checkout(j) for j in range(1, 4)])
    return {
        'id': 'retail_stores',
        'name': 'Retail store network',
        'category': 'business',
        'category_label': 'Business Operations',
        'icon': 'mdi-storefront',
        'description': '3 stores, 3 checkouts each. Foot traffic, transaction counts, and queue times feed anomaly detection for staffing decisions.',
        'config': {
            'level_names': ['company', 'region', 'store', 'checkout'],
            'root_name': 'company',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('company', children=[
            _node('region_north', children=[store(i) for i in range(1, 4)]),
        ]),
    }


def _restaurant_chain():
    def equipment(i):
        return _node(f'kitchen_equipment_{i}', children=[
            _sensor('temperature', 'celsius', 1),
            _sensor('usage_count', 'custom', 1, 'int'),
            _sensor('power_draw', 'w', 10),
        ])
    def location(i):
        return _node(f'location_{i}', children=[equipment(j) for j in range(1, 3)])
    return {
        'id': 'restaurant_chain',
        'name': 'Restaurant chain',
        'category': 'business',
        'category_label': 'Business Operations',
        'icon': 'mdi-silverware-fork-knife',
        'description': '3 locations, 2 kitchen equipment units each (fryer + oven, etc.). Temperature, usage count, and power draw for equipment health monitoring.',
        'config': {
            'level_names': ['chain', 'region', 'location', 'equipment'],
            'root_name': 'chain',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('chain', children=[
            _node('region_central', children=[location(i) for i in range(1, 4)]),
        ]),
    }


def _pharmacy_coldchain():
    def refrigerator(i):
        return _node(f'refrigerator_{i}', children=[
            _sensor('temperature', 'celsius', 1),
            _sensor('door_open_count', 'custom', 1, 'int'),
            _sensor('compressor_current', 'a', 10),
        ])
    def store(i):
        return _node(f'store_{i}', children=[refrigerator(j) for j in range(1, 3)])
    return {
        'id': 'pharmacy_coldchain',
        'name': 'Pharmacy cold-chain',
        'category': 'business',
        'category_label': 'Business Operations',
        'icon': 'mdi-medical-bag',
        'description': '3 stores, 2 vaccine refrigerators each. Temperature, door-open counts, and compressor current — cold-chain compliance monitoring.',
        'config': {
            'level_names': ['chain', 'region', 'store', 'refrigerator'],
            'root_name': 'chain',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('chain', children=[
            _node('region_east', children=[store(i) for i in range(1, 4)]),
        ]),
    }


# ── Facilities & Infrastructure ────────────────────────────────────────────

def _data_center():
    def server(i):
        return _node(f'server_{i}', children=[
            _sensor('cpu_temp', 'celsius', 1),
            _sensor('power_draw', 'w', 1),
            _sensor('fan_rpm', 'rpm', 1),
        ])
    def rack(i):
        return _node(f'rack_{i}', children=[server(j) for j in range(1, 3)])
    def row(i):
        return _node(f'row_{i}', children=[rack(j) for j in range(1, 4)])
    return {
        'id': 'data_center',
        'name': 'Data center monitoring',
        'category': 'facilities',
        'category_label': 'Facilities & Infrastructure',
        'icon': 'mdi-server',
        'description': '2 rows, 3 racks each, 2 servers per rack. CPU temperature, power draw, and fan RPM — a great fit for thermal-anomaly / failure-prediction models.',
        'config': {
            'level_names': ['datacenter', 'row', 'rack', 'server'],
            'root_name': 'datacenter',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('datacenter', children=[row(i) for i in range(1, 3)]),
    }


def _warehouse():
    def conveyor(i):
        return _node(f'conveyor_{i}', children=[
            _sensor('belt_speed', 'custom', 10, 'float'),  # m/s
            _sensor('package_count', 'custom', 1, 'int'),
            _sensor('weight', 'custom', 10, 'float'),  # kg
        ])
    def zone(i):
        return _node(f'zone_{i}', children=[conveyor(j) for j in range(1, 3)])
    return {
        'id': 'warehouse',
        'name': 'Warehouse operations',
        'category': 'facilities',
        'category_label': 'Facilities & Infrastructure',
        'icon': 'mdi-warehouse',
        'description': '1 warehouse, 3 zones, 2 conveyors per zone. Belt speed, package count, and weight for throughput and jam-detection models.',
        'config': {
            'level_names': ['company', 'warehouse', 'zone', 'conveyor'],
            'root_name': 'company',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('company', children=[
            _node('warehouse_A', children=[zone(i) for i in range(1, 4)]),
        ]),
    }


def _office_hvac():
    def hvac(i):
        return _node(f'hvac_unit_{i}', children=[
            _sensor('temperature', 'celsius', 1),
            _sensor('humidity', 'percent', 1),
            _sensor('air_quality', 'custom', 1, 'float'),  # CO2 ppm typically
        ])
    def zone(i):
        return _node(f'zone_{i}', children=[hvac(j) for j in range(1, 3)])
    def floor(i):
        return _node(f'floor_{i}', children=[zone(j) for j in range(1, 3)])
    return {
        'id': 'office_hvac',
        'name': 'Office building HVAC',
        'category': 'facilities',
        'category_label': 'Facilities & Infrastructure',
        'icon': 'mdi-office-building',
        'description': '3 floors, 2 zones per floor, 2 HVAC units per zone. Temperature, humidity, air quality — comfort optimization and equipment health.',
        'config': {
            'level_names': ['building', 'floor', 'zone', 'hvac'],
            'root_name': 'building',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('building', children=[floor(i) for i in range(1, 4)]),
    }


# ── Energy & Environment ───────────────────────────────────────────────────

def _solar_farm():
    def panel(i):
        return _node(f'panel_{i}', children=[
            _sensor('voltage', 'v', 1),
            _sensor('current', 'a', 1),
            _sensor('temperature', 'celsius', 1),
            _sensor('irradiance', 'custom', 1, 'float'),  # W/m²
        ])
    def inverter(i):
        return _node(f'inverter_{i}', children=[panel(j) for j in range(1, 5)])
    def array(i):
        return _node(f'array_{i}', children=[inverter(1)])
    return {
        'id': 'solar_farm',
        'name': 'Solar farm',
        'category': 'energy',
        'category_label': 'Energy & Environment',
        'icon': 'mdi-solar-power',
        'description': '3 arrays, each with 1 inverter and 4 panels. Voltage, current, temperature, and irradiance for panel-fault detection and yield forecasting.',
        'config': {
            'level_names': ['farm', 'array', 'inverter', 'panel'],
            'root_name': 'farm',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('farm', children=[array(i) for i in range(1, 4)]),
    }


def _water_treatment():
    def tank(i):
        return _node(f'tank_{i}', children=[
            _sensor('ph', 'custom', 1, 'float'),
            _sensor('turbidity', 'custom', 1, 'float'),  # NTU
            _sensor('flow_rate', 'custom', 1, 'float'),  # L/s
            _sensor('temperature', 'celsius', 1),
        ])
    return {
        'id': 'water_treatment',
        'name': 'Water treatment plant',
        'category': 'energy',
        'category_label': 'Energy & Environment',
        'icon': 'mdi-water',
        'description': '1 plant, 3 tanks. pH, turbidity, flow rate, and temperature — compliance monitoring and process anomaly detection.',
        'config': {
            'level_names': ['utility', 'plant', 'tank', 'probe'],
            'root_name': 'utility',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('utility', children=[
            _node('plant_A', children=[tank(i) for i in range(1, 4)]),
        ]),
    }


# ── Healthcare ─────────────────────────────────────────────────────────────

def _hospital():
    def vitals(_i):
        return _node('vitals_monitor', children=[
            _sensor('heart_rate', 'custom', 1, 'int'),  # bpm
            _sensor('spo2', 'percent', 1, 'int'),
            _sensor('temperature', 'celsius', 1),
            _sensor('respiratory_rate', 'custom', 1, 'int'),  # breaths / min
        ])
    def bed(i):
        return _node(f'bed_{i}', children=[vitals(i)])
    def ward(i):
        return _node(f'ward_{i}', children=[bed(j) for j in range(1, 5)])
    return {
        'id': 'hospital',
        'name': 'Hospital patient monitoring',
        'category': 'healthcare',
        'category_label': 'Healthcare',
        'icon': 'mdi-hospital-building',
        'description': '2 wards, 4 beds each, one vitals monitor per bed. Heart rate, SpO2, temperature, respiratory rate — early-warning score models.',
        'config': {
            'level_names': ['hospital', 'ward', 'bed', 'monitor', 'sensor'],
            'root_name': 'hospital',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('hospital', children=[ward(i) for i in range(1, 3)]),
    }


# ── Blank ──────────────────────────────────────────────────────────────────

def _blank():
    return {
        'id': 'blank',
        'name': 'Blank starter',
        'category': 'blank',
        'category_label': 'Blank',
        'icon': 'mdi-file-outline',
        'description': 'Just the root node. Use this if you want to skip the level-names step but build the tree yourself.',
        'config': {
            'level_names': ['factory', 'plant', 'machine', 'sensor'],
            'root_name': 'factory',
            'topic_mode': 'strict',
            'meta_prefixes': DEFAULT_META,
        },
        'tree': _node('factory', children=[]),
    }


# ── Assembly ───────────────────────────────────────────────────────────────

TREE_TEMPLATES = [
    _small_factory(),
    _vibration_rig(),
    _rotating_fleet(),
    _retail_stores(),
    _restaurant_chain(),
    _pharmacy_coldchain(),
    _data_center(),
    _warehouse(),
    _office_hvac(),
    _solar_farm(),
    _water_treatment(),
    _hospital(),
    _blank(),
]


def get_all():
    """Return the full list — used by GET /tree-templates."""
    return TREE_TEMPLATES


def get_by_id(template_id):
    """Return one template by id, or None."""
    for t in TREE_TEMPLATES:
        if t['id'] == template_id:
            return t
    return None
