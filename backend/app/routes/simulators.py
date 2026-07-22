"""CiRA ME — Machine Simulator Routes (Phase F, 2026-07-19).

REST facade over `services/machine_simulator.py`. Mirrors the auth
pattern used by `routes/asset_tree.py`:
- Read endpoints (list, snapshot, profiles) allow any logged-in user.
- Write endpoints (create/patch/delete + publish-raw) are admin-only.

Every mutation is logged to `asset_tree_audit` so operators have a
paper trail of who spun up / tore down / poked what.

See docs/PLAN_2026-07-19_machine-simulator.md §4.2 for the spec.
"""

import base64
import binascii
import logging

from flask import Blueprint, request, jsonify

from ..auth import login_required
from ..models import AssetTreeAudit
from ..services.machine_simulator import machine_simulator

logger = logging.getLogger(__name__)
simulators_bp = Blueprint('simulators', __name__)


def _admin_only():
    """Return 403 JSON response if current user isn't admin, else None.
    Duplicated from routes/asset_tree.py so importing the constants
    module doesn't create a hard dependency cycle. Kept identical."""
    user = getattr(request, 'current_user', None)
    if not user or user.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    return None


def _actor_id() -> int:
    """Return current user id or 0 (sentinel for system / anon)."""
    u = getattr(request, 'current_user', None) or {}
    return int(u.get('id') or 0)


# ── Read endpoints ────────────────────────────────────────────────────────


@simulators_bp.route('/profiles', methods=['GET'])
@login_required
def list_profiles():
    return jsonify({'profiles': machine_simulator.list_profiles()})


@simulators_bp.route('/', methods=['GET'])
@simulators_bp.route('', methods=['GET'])
@login_required
def list_instances():
    return jsonify({'instances': machine_simulator.list_instances()})


@simulators_bp.route('/snapshot', methods=['GET'])
@login_required
def get_snapshot():
    return jsonify(machine_simulator.snapshot())


# ── Write endpoints (admin only) ─────────────────────────────────────────


@simulators_bp.route('/', methods=['POST'])
@simulators_bp.route('', methods=['POST'])
@login_required
def create_instance():
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    profile_id = data.get('profile_id')
    name = data.get('name')
    topic_base = data.get('topic_base')
    initial_state = data.get('initial_state')
    autoprovision = bool(data.get('autoprovision_tree', False))

    try:
        instance = machine_simulator.create_instance(
            profile_id=profile_id,
            name=name,
            topic_base=topic_base,
            initial_state=initial_state,
            autoprovision_tree=autoprovision,
            actor_user_id=_actor_id(),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('create_instance failed')
        return jsonify({'error': str(e)}), 500

    try:
        AssetTreeAudit.log(
            actor_user_id=_actor_id(),
            event_type='simulator_create',
            target_type='simulator',
            target_id=None,
            payload={
                'instance_id': instance.get('id'),
                'profile_id': profile_id,
                'name': name,
                'topic_base': topic_base,
                'initial_state': instance.get('state'),
                'autoprovision_tree': autoprovision,
                'autoprovisioned_node_ids': instance.get(
                    'autoprovisioned_node_ids', []),
            },
        )
    except Exception:
        logger.warning('[sim] audit log create failed', exc_info=True)

    return jsonify(instance), 201


@simulators_bp.route('/<instance_id>', methods=['PATCH'])
@login_required
def patch_instance(instance_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    new_state = data.get('state')
    if not new_state or not isinstance(new_state, str):
        return jsonify({'error': "'state' (string) required"}), 400
    try:
        instance = machine_simulator.patch_state(
            instance_id=instance_id,
            new_state=new_state,
            actor_user_id=_actor_id(),
        )
    except KeyError:
        return jsonify({'error': f'Instance {instance_id!r} not found'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('patch_instance failed')
        return jsonify({'error': str(e)}), 500

    try:
        AssetTreeAudit.log(
            actor_user_id=_actor_id(),
            event_type='simulator_patch_state',
            target_type='simulator',
            target_id=None,
            payload={
                'instance_id': instance_id,
                'name': instance.get('name'),
                'new_state': new_state,
            },
        )
    except Exception:
        logger.warning('[sim] audit log patch failed', exc_info=True)

    return jsonify(instance)


@simulators_bp.route('/<instance_id>/change-profile', methods=['POST'])
@login_required
def change_profile(instance_id):
    """Swap a running simulator's profile without deleting/recreating.

    Body:
        {
          "profile_id": "industrial_boiler",   # required
          "state": "idle"                      # optional; defaults to new
                                               # profile's default_state
        }

    Stops the sim thread, retires the current profile's sensor children
    under the machine node, autoprovisions the new profile's sensors,
    and restarts the thread on the new profile at `state`. The audit event
    `simulator_change_profile` records what got retired + created.

    Phase G — Q1 (2026-07-22 spec).
    """
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    profile_id = data.get('profile_id')
    if not profile_id or not isinstance(profile_id, str):
        return jsonify({'error': "'profile_id' (string) required"}), 400
    new_state = data.get('state')
    if new_state is not None and not isinstance(new_state, str):
        return jsonify({'error': "'state' must be a string when provided"}), 400

    try:
        instance = machine_simulator.change_profile(
            instance_id=instance_id,
            new_profile_id=profile_id,
            new_state=new_state,
            actor_user_id=_actor_id(),
        )
    except KeyError:
        return jsonify({'error': f'Instance {instance_id!r} not found'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        # Broker down / router unavailable → 503, matches publish-raw pattern.
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.exception('change_profile failed')
        return jsonify({'error': str(e)}), 500

    # (Audit event is emitted inside machine_simulator.change_profile so the
    # payload can reference the exact retired/created node ids the routine
    # observed. Route-level catch-all here is intentionally omitted.)
    return jsonify(instance)


@simulators_bp.route('/<instance_id>', methods=['DELETE'])
@login_required
def delete_instance(instance_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    # Snapshot name before deletion so the audit row is readable.
    with machine_simulator._instances_lock:  # noqa — private access is fine within the same package
        m = machine_simulator._instances.get(instance_id)
        name = m.name if m else None
    try:
        machine_simulator.delete_instance(
            instance_id=instance_id, actor_user_id=_actor_id(),
        )
    except KeyError:
        return jsonify({'error': f'Instance {instance_id!r} not found'}), 404
    except Exception as e:
        logger.exception('delete_instance failed')
        return jsonify({'error': str(e)}), 500

    try:
        AssetTreeAudit.log(
            actor_user_id=_actor_id(),
            event_type='simulator_delete',
            target_type='simulator',
            target_id=None,
            payload={'instance_id': instance_id, 'name': name},
        )
    except Exception:
        logger.warning('[sim] audit log delete failed', exc_info=True)

    return '', 204


@simulators_bp.route('/publish-raw', methods=['POST'])
@login_required
def publish_raw():
    """One-shot arbitrary MQTT publish.

    Body shape (any one of):
      { "topic": "...", "payload": "..." }          → utf-8 encoded
      { "topic": "...", "payload_hex": "0001ff" }   → raw bytes via hex
      { "topic": "...", "payload_b64": "..." }      → base64 bytes
    """
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    topic = data.get('topic')
    if not topic or not isinstance(topic, str):
        return jsonify({'error': "'topic' (string) required"}), 400

    # Reject empty payloads up-front — otherwise they publish 0 bytes and
    # the ingest router silently logs "unparseable payload", surfacing no
    # error to the user. (QA F.QA polish #7.)
    has_hex = 'payload_hex' in data and data['payload_hex']
    has_b64 = 'payload_b64' in data and data['payload_b64']
    has_payload = 'payload' in data and data['payload'] not in (None, '')
    if not (has_hex or has_b64 or has_payload):
        return jsonify({
            'error': 'payload required (non-empty payload, payload_hex, '
                     'or payload_b64)'
        }), 400

    if 'payload_hex' in data and data['payload_hex']:
        try:
            payload_bytes = binascii.unhexlify(
                str(data['payload_hex']).replace(' ', '').replace('\\x', '')
            )
        except (binascii.Error, ValueError):
            return jsonify({'error': 'payload_hex is not valid hex'}), 400
    elif 'payload_b64' in data and data['payload_b64']:
        try:
            payload_bytes = base64.b64decode(str(data['payload_b64']),
                                             validate=True)
        except (binascii.Error, ValueError):
            return jsonify({'error': 'payload_b64 is not valid base64'}), 400
    else:
        payload = data.get('payload')
        if payload is None:
            payload = ''
        if not isinstance(payload, (str, bytes)):
            payload = str(payload)
        # Support "\x00\x01" style escapes as a convenience for the raw-widget.
        if isinstance(payload, str) and '\\x' in payload:
            try:
                payload_bytes = payload.encode('utf-8').decode(
                    'unicode_escape').encode('latin-1', errors='replace')
            except Exception:
                payload_bytes = payload.encode('utf-8', errors='replace')
        elif isinstance(payload, str):
            payload_bytes = payload.encode('utf-8', errors='replace')
        else:
            payload_bytes = payload

    try:
        machine_simulator.publish_raw(
            topic=topic, payload_bytes=payload_bytes,
            actor_user_id=_actor_id(),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        # Broker down — return 503 so the UI knows to retry later.
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.exception('publish_raw failed')
        return jsonify({'error': str(e)}), 500

    try:
        AssetTreeAudit.log(
            actor_user_id=_actor_id(),
            event_type='simulator_publish_raw',
            target_type='simulator',
            target_id=None,
            payload={'topic': topic, 'payload_length': len(payload_bytes)},
        )
    except Exception:
        logger.warning('[sim] audit log publish-raw failed', exc_info=True)

    return jsonify({
        'topic': topic,
        'payload_length': len(payload_bytes),
        'ok': True,
    })
