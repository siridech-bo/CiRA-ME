"""CiRA ME — Machine Profile Library (Phase F, 2026-07-19).

Static catalog of realistic machine simulator profiles. Each profile
defines the sensors present on the machine, the states it can be in,
and per-state signal-generation parameters used by
`services/machine_simulator.py`.

Design ref: docs/PLAN_2026-07-19_machine-simulator.md §3.

State parameters are `(mean, std_dev, sinusoid_amplitude, sinusoid_period_s)`.
A tuple of `None` in `per_sensor` means the sensor stays silent in that
state (e.g. "off" or "maintenance" — no publish at all). A special
`'chaos'` state ships with every profile and is handled specially by
the simulator service (see §3.7): normal signal generation runs on
top of periodic poison-message injection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random


# ── Types ────────────────────────────────────────────────────────────────


@dataclass
class SensorDef:
    name: str            # topic segment (regex ^[A-Za-z0-9_-]+$)
    unit: str            # 'bar', 'celsius', 'mm_s2', 'a', 'hz', 'percent'
    sample_rate_hz: float  # publish rate (Hz)


# Params tuple: (mean, std_dev, sinusoid_amplitude, sinusoid_period_s)
# When sinusoid_amplitude == 0 the signal is pure Gaussian around mean.
SensorParams = Optional[Tuple[float, float, float, float]]


@dataclass
class StateParams:
    # Per-sensor: params tuple, or None if the sensor is silent this state.
    per_sensor: Dict[str, SensorParams]
    # Optional fault spike probability per tick per sensor (0..1). Currently
    # unused by the simulator (Gaussian noise + sinusoid captures the vibes
    # for the demo), but the field stays here so we can wire in punch-spikes
    # in a follow-up without changing the profile schema.
    fault_spikes: Dict[str, float] = field(default_factory=dict)


@dataclass
class MachineProfile:
    id: str
    display_name: str
    icon: str              # mdi- name
    description: str
    sensors: List[SensorDef]
    states: Dict[str, StateParams]  # keys: 'off', 'idle', 'running', ...
    default_state: str

    def to_dict(self) -> dict:
        """JSON-safe projection for the /profiles endpoint."""
        return {
            'id': self.id,
            'display_name': self.display_name,
            'icon': self.icon,
            'description': self.description,
            'sensors': [
                {
                    'name': s.name,
                    'unit': s.unit,
                    'sample_rate_hz': s.sample_rate_hz,
                }
                for s in self.sensors
            ],
            'states': sorted(list(self.states.keys())),
            'default_state': self.default_state,
        }


# ── Signal generator ─────────────────────────────────────────────────────

def sample(
    state_params: StateParams,
    sensor_name: str,
    t_seconds: float,
) -> Optional[float]:
    """Return a sample value for `sensor_name` under `state_params` at
    monotonic time `t_seconds`, or None if silent in this state.

    Signal model: sinusoid + Gaussian noise around `mean`. If
    `sinusoid_amplitude == 0`, drops the sinusoid term.
    """
    if state_params is None:
        return None
    params = state_params.per_sensor.get(sensor_name)
    if params is None:
        return None
    mean, std, amp, period = params
    # Guard tiny/zero periods so a bad table entry doesn't div-by-zero.
    if amp != 0 and period > 0:
        sine = amp * math.sin(2.0 * math.pi * (t_seconds / period))
    else:
        sine = 0.0
    noise = random.gauss(0.0, std) if std > 0 else 0.0
    return mean + sine + noise


# ── Common shorthand ─────────────────────────────────────────────────────
# A silent state (used for "off" / "maintenance"): every sensor param is None.
def _silent(sensors: List[SensorDef]) -> StateParams:
    return StateParams(per_sensor={s.name: None for s in sensors})


# Convenient tuple builder for tables: (mean, std) -> pure noise.
def _n(mean: float, std: float) -> SensorParams:
    return (float(mean), float(std), 0.0, 0.0)


# Sinusoid + noise: (mean, std, amp, period_s).
def _s(mean: float, std: float, amp: float, period: float) -> SensorParams:
    return (float(mean), float(std), float(amp), float(period))


# ── 3.1 Air Compressor ───────────────────────────────────────────────────

_COMPRESSOR_SENSORS = [
    SensorDef('pressure', 'bar', 1.0),
    SensorDef('temperature', 'celsius', 1.0),
    SensorDef('vibration', 'mm_s2', 5.0),
    SensorDef('current', 'a', 1.0),
]

_AIR_COMPRESSOR = MachineProfile(
    id='air_compressor',
    display_name='Air Compressor',
    icon='mdi-fan',
    description='Reciprocating air compressor — pressure, temp, vibration, current.',
    sensors=_COMPRESSOR_SENSORS,
    default_state='running',
    states={
        'idle': StateParams(per_sensor={
            'pressure':    _n(1.0, 0.05),
            'temperature': _n(25.0, 1.0),
            'vibration':   _n(0.3, 0.1),
            'current':     _n(0.5, 0.2),
        }),
        'running': StateParams(per_sensor={
            'pressure':    _s(4.0, 0.2, 2.0, 30.0),  # 4.0 + 2.0*sin(30s)
            'temperature': _n(55.0, 3.0),
            'vibration':   _n(2.5, 0.4),
            'current':     _n(15.0, 1.0),
        }),
        'loaded': StateParams(per_sensor={
            'pressure':    _n(6.8, 0.3),
            'temperature': _n(72.0, 3.0),
            'vibration':   _n(3.2, 0.5),
            'current':     _n(22.0, 1.5),
        }),
        'fault': StateParams(per_sensor={
            'pressure':    _n(3.5, 0.5),
            'temperature': _n(88.0, 4.0),
            'vibration':   _n(7.5, 1.2),   # bearing fault
            'current':     _n(27.0, 2.0),
        }),
        'maintenance': _silent(_COMPRESSOR_SENSORS),
        # chaos state: still publishes normal running signals but the
        # simulator injects poison messages on top. Signal params match
        # 'running' so operators can see clean+chaos overlay in sparklines.
        'chaos': StateParams(per_sensor={
            'pressure':    _s(4.0, 0.2, 2.0, 30.0),
            'temperature': _n(55.0, 3.0),
            'vibration':   _n(2.5, 0.4),
            'current':     _n(15.0, 1.0),
        }),
    },
)


# ── 3.2 Industrial Boiler ────────────────────────────────────────────────

_BOILER_SENSORS = [
    SensorDef('flame', 'percent', 0.5),
    SensorDef('feedwater_temp', 'celsius', 1.0),
    SensorDef('steam_pressure', 'bar', 1.0),
    SensorDef('flue_gas_temp', 'celsius', 1.0),
    SensorDef('o2_percent', 'percent', 1.0),
]

_BOILER = MachineProfile(
    id='industrial_boiler',
    display_name='Industrial Boiler',
    icon='mdi-radiator',
    description='Steam boiler — flame, feedwater/flue temps, O2, steam pressure.',
    sensors=_BOILER_SENSORS,
    default_state='steaming',
    states={
        'off': StateParams(per_sensor={
            'flame':          _n(0.0, 0.0),
            'feedwater_temp': _n(25.0, 0.3),
            'steam_pressure': _n(0.0, 0.01),
            'flue_gas_temp':  _n(25.0, 0.3),
            'o2_percent':     _n(20.9, 0.05),
        }),
        # Warming: flame duty ~70% (approximated as noisy 0.7 signal),
        # temps ramp — represented as their midpoint here (state changes
        # per-tick ramping is out of scope for MVP).
        'warming': StateParams(per_sensor={
            'flame':          _n(0.7, 0.15),
            'feedwater_temp': _n(52.0, 8.0),
            'steam_pressure': _n(1.0, 0.4),
            'flue_gas_temp':  _n(100.0, 25.0),
            'o2_percent':     _n(5.0, 1.0),
        }),
        'steaming': StateParams(per_sensor={
            'flame':          _n(0.5, 0.1),
            'feedwater_temp': _n(85.0, 3.0),
            'steam_pressure': _s(8.0, 0.15, 0.5, 60.0),  # 8 + 0.5*sin(60s)
            'flue_gas_temp':  _n(220.0, 15.0),
            'o2_percent':     _n(4.0, 0.5),
        }),
        'blowdown': StateParams(per_sensor={
            'flame':          _n(0.0, 0.0),
            'feedwater_temp': _n(72.0, 8.0),   # midpoint of 85→60
            'steam_pressure': _n(5.5, 1.5),
            'flue_gas_temp':  _n(185.0, 20.0),
            'o2_percent':     _n(20.9, 0.05),
        }),
        'fault': StateParams(per_sensor={
            'flame':          _n(1.0, 0.02),
            'feedwater_temp': _n(95.0, 2.0),
            'steam_pressure': _n(12.0, 0.4),   # over-pressure
            'flue_gas_temp':  _n(320.0, 8.0),  # unsafe
            'o2_percent':     _n(1.0, 0.3),    # rich
        }),
        'chaos': StateParams(per_sensor={
            'flame':          _n(0.5, 0.1),
            'feedwater_temp': _n(85.0, 3.0),
            'steam_pressure': _s(8.0, 0.15, 0.5, 60.0),
            'flue_gas_temp':  _n(220.0, 15.0),
            'o2_percent':     _n(4.0, 0.5),
        }),
    },
)


# ── 3.3 Centrifugal Pump ─────────────────────────────────────────────────

_PUMP_SENSORS = [
    SensorDef('inlet_p', 'bar', 1.0),
    SensorDef('outlet_p', 'bar', 1.0),
    SensorDef('flow_rate', 'lpm', 1.0),
    SensorDef('motor_current', 'a', 1.0),
    SensorDef('vibration', 'mm_s2', 5.0),
]

_PUMP = MachineProfile(
    id='centrifugal_pump',
    display_name='Centrifugal Pump',
    icon='mdi-pump',
    description='Water pump — inlet/outlet pressure, flow, motor current, vibration.',
    sensors=_PUMP_SENSORS,
    default_state='running',
    states={
        'off': StateParams(per_sensor={
            'inlet_p':       _n(1.0, 0.05),
            'outlet_p':      _n(1.0, 0.05),
            'flow_rate':     _n(0.0, 0.05),
            'motor_current': _n(0.3, 0.05),
            'vibration':     _n(0.2, 0.05),
        }),
        'running': StateParams(per_sensor={
            'inlet_p':       _n(1.5, 0.1),
            'outlet_p':      _n(4.5, 0.2),
            'flow_rate':     _n(120.0, 10.0),
            'motor_current': _n(8.0, 0.5),
            'vibration':     _n(1.8, 0.3),
        }),
        'cavitation': StateParams(per_sensor={
            'inlet_p':       _n(0.4, 0.3),
            'outlet_p':      _n(3.5, 0.8),
            'flow_rate':     _n(90.0, 25.0),
            'motor_current': _n(9.5, 1.0),
            'vibration':     _n(5.5, 1.5),
        }),
        'dry_run': StateParams(per_sensor={
            'inlet_p':       _n(0.9, 0.05),
            'outlet_p':      _n(1.1, 0.1),
            'flow_rate':     _n(0.0, 2.0),
            'motor_current': _n(6.0, 0.5),
            'vibration':     _n(4.5, 0.8),
        }),
        'chaos': StateParams(per_sensor={
            'inlet_p':       _n(1.5, 0.1),
            'outlet_p':      _n(4.5, 0.2),
            'flow_rate':     _n(120.0, 10.0),
            'motor_current': _n(8.0, 0.5),
            'vibration':     _n(1.8, 0.3),
        }),
    },
)


# ── 3.4 Conveyor ─────────────────────────────────────────────────────────

_CONVEYOR_SENSORS = [
    SensorDef('speed_rpm', 'rpm', 1.0),
    SensorDef('motor_current', 'a', 1.0),
    SensorDef('belt_tension', 'kn', 1.0),
    SensorDef('vibration', 'mm_s2', 5.0),
    SensorDef('product_count', 'count', 0.2),
]

_CONVEYOR = MachineProfile(
    id='conveyor',
    display_name='Conveyor',
    icon='mdi-package-variant-closed',
    description='Belt conveyor — speed, current, belt tension, vibration, product count.',
    sensors=_CONVEYOR_SENSORS,
    default_state='running',
    states={
        'off': StateParams(per_sensor={
            'speed_rpm':      _n(0.0, 0.5),
            'motor_current':  _n(0.3, 0.05),
            'belt_tension':   _n(2.0, 0.2),
            'vibration':      _n(0.1, 0.05),
            'product_count':  _n(0.0, 0.0),
        }),
        'running': StateParams(per_sensor={
            'speed_rpm':      _n(900.0, 20.0),
            'motor_current':  _n(4.0, 0.5),
            'belt_tension':   _n(3.5, 0.2),
            'vibration':      _n(1.5, 0.3),
            # product_count trickles up; simulator emits fractional counts.
            'product_count':  _n(12.0, 2.0),
        }),
        'jam': StateParams(per_sensor={
            'speed_rpm':      _n(0.0, 0.3),
            'motor_current':  _n(12.0, 0.5),   # motor stall
            'belt_tension':   _n(6.0, 0.4),    # over-tension
            'vibration':      _n(3.5, 0.7),
            'product_count':  _n(0.0, 0.0),
        }),
        'belt_slip': StateParams(per_sensor={
            'speed_rpm':      _n(900.0, 25.0),  # motor rpm normal
            'motor_current':  _n(6.0, 1.0),
            'belt_tension':   _n(1.8, 0.5),
            'vibration':      _n(4.2, 0.9),
            'product_count':  _n(3.0, 2.0),
        }),
        'chaos': StateParams(per_sensor={
            'speed_rpm':      _n(900.0, 20.0),
            'motor_current':  _n(4.0, 0.5),
            'belt_tension':   _n(3.5, 0.2),
            'vibration':      _n(1.5, 0.3),
            'product_count':  _n(12.0, 2.0),
        }),
    },
)


# ── 3.5 CNC Spindle ──────────────────────────────────────────────────────

_CNC_SENSORS = [
    SensorDef('spindle_rpm', 'rpm', 1.0),
    SensorDef('spindle_load', 'percent', 1.0),
    SensorDef('temperature', 'celsius', 1.0),
    SensorDef('vibration_x', 'mm_s2', 10.0),
    SensorDef('vibration_y', 'mm_s2', 10.0),
    SensorDef('vibration_z', 'mm_s2', 10.0),
]

_CNC = MachineProfile(
    id='cnc_spindle',
    display_name='CNC Spindle',
    icon='mdi-cog',
    description='CNC machining spindle — rpm, load, temp, tri-axial vibration.',
    sensors=_CNC_SENSORS,
    default_state='cutting',
    states={
        'off': StateParams(per_sensor={
            'spindle_rpm':   _n(0.0, 0.5),
            'spindle_load':  _n(0.0, 0.1),
            'temperature':   _n(22.0, 1.0),
            'vibration_x':   _n(0.05, 0.02),
            'vibration_y':   _n(0.05, 0.02),
            'vibration_z':   _n(0.05, 0.02),
        }),
        'idle': StateParams(per_sensor={
            'spindle_rpm':   _n(800.0, 20.0),
            'spindle_load':  _n(5.0, 1.0),
            'temperature':   _n(30.0, 2.0),
            'vibration_x':   _n(0.4, 0.1),
            'vibration_y':   _n(0.4, 0.1),
            'vibration_z':   _n(0.4, 0.1),
        }),
        'cutting': StateParams(per_sensor={
            'spindle_rpm':   _s(6000.0, 40.0, 400.0, 15.0),  # 6000 + 400·sin(15s)
            'spindle_load':  _n(55.0, 8.0),
            'temperature':   _n(45.0, 3.0),
            'vibration_x':   _n(1.8, 0.3),
            'vibration_y':   _n(2.1, 0.3),
            'vibration_z':   _n(1.5, 0.3),
        }),
        'chatter': StateParams(per_sensor={
            'spindle_rpm':   _n(6000.0, 200.0),
            'spindle_load':  _n(75.0, 15.0),
            'temperature':   _n(52.0, 4.0),
            'vibration_x':   _n(6.5, 1.2),
            'vibration_y':   _n(8.2, 1.5),
            'vibration_z':   _n(5.0, 1.0),
        }),
        'chaos': StateParams(per_sensor={
            'spindle_rpm':   _s(6000.0, 40.0, 400.0, 15.0),
            'spindle_load':  _n(55.0, 8.0),
            'temperature':   _n(45.0, 3.0),
            'vibration_x':   _n(1.8, 0.3),
            'vibration_y':   _n(2.1, 0.3),
            'vibration_z':   _n(1.5, 0.3),
        }),
    },
)


# ── 3.6 Chiller / HVAC ───────────────────────────────────────────────────

_CHILLER_SENSORS = [
    SensorDef('refrigerant_p', 'bar', 1.0),
    SensorDef('evap_temp', 'celsius', 1.0),
    SensorDef('cond_temp', 'celsius', 1.0),
    SensorDef('compressor_current', 'a', 1.0),
]

_CHILLER = MachineProfile(
    id='chiller_hvac',
    display_name='Chiller / HVAC',
    icon='mdi-snowflake',
    description='Refrigeration chiller — refrigerant pressure, evap/cond temps, current.',
    sensors=_CHILLER_SENSORS,
    default_state='cooling',
    states={
        'off': StateParams(per_sensor={
            'refrigerant_p':      _n(5.0, 0.1),
            'evap_temp':          _n(25.0, 1.0),
            'cond_temp':          _n(25.0, 1.0),
            'compressor_current': _n(0.0, 0.05),
        }),
        'cooling': StateParams(per_sensor={
            'refrigerant_p':      _n(8.5, 0.3),
            'evap_temp':          _n(-2.0, 1.0),
            'cond_temp':          _n(42.0, 2.0),
            'compressor_current': _n(14.0, 0.5),
        }),
        'defrost': StateParams(per_sensor={
            'refrigerant_p':      _n(6.0, 0.2),
            'evap_temp':          _n(10.0, 2.0),
            'cond_temp':          _n(30.0, 2.0),
            'compressor_current': _n(3.0, 0.3),
        }),
        'fault': StateParams(per_sensor={
            'refrigerant_p':      _n(12.0, 0.2),
            'evap_temp':          _n(15.0, 0.5),   # no cooling
            'cond_temp':          _n(65.0, 1.0),
            'compressor_current': _n(20.0, 0.5),
        }),
        'chaos': StateParams(per_sensor={
            'refrigerant_p':      _n(8.5, 0.3),
            'evap_temp':          _n(-2.0, 1.0),
            'cond_temp':          _n(42.0, 2.0),
            'compressor_current': _n(14.0, 0.5),
        }),
    },
)


# ── Registry ─────────────────────────────────────────────────────────────

_PROFILES: Dict[str, MachineProfile] = {
    p.id: p for p in [
        _AIR_COMPRESSOR,
        _BOILER,
        _PUMP,
        _CONVEYOR,
        _CNC,
        _CHILLER,
    ]
}


def get_all_profiles() -> List[MachineProfile]:
    """List every profile in the catalog. Ordered by insertion above."""
    return list(_PROFILES.values())


def get_profile(profile_id: str) -> Optional[MachineProfile]:
    return _PROFILES.get(profile_id)
