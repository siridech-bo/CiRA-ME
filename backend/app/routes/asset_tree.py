"""CiRA ME - Asset Tree Routes (Phase A, 2026-07-18).

Physical-asset hierarchy (Factory / Plant / Machine / Sensor by default)
replacing the "Project" abstraction. Strictly additive — does NOT touch
existing endpoint contracts.

Reads: any logged-in user.
Writes: admin only. Every state-changing op is audited.

See:
- docs/PLAN_2026-07-18_asset-tree.md — full design
- docs/EXECUTION_asset-tree.md       — subtask IDs
"""

import re
import logging
from flask import Blueprint, request, jsonify

from ..auth import login_required
from ..models import (
    AssetTreeConfig, AssetNode, AssetSensorMeta,
    MachineGroup, AssetTreeAudit, ModelMachineBinding, SavedModel,
    get_db,
)
from ..constants.sensor_presets import (
    UNIT_PRESETS, SAMPLE_RATE_PRESETS,
    SENSOR_TEMPLATES, HIERARCHY_PRESETS,
)

logger = logging.getLogger(__name__)
asset_tree_bp = Blueprint('asset_tree', __name__)


def _reload_ingest_router():
    """Phase D — notify the ingest router that the tree changed.

    Every tree-mutation route calls this at end-of-request so newly-created
    / renamed / retired paths hit the router's in-memory cache without a
    poll. Fire-and-forget by design; failures are swallowed (router runs
    a periodic safety-net refresh anyway) so a broken import can't 500
    the underlying admin action.
    """
    try:
        from ..services.mqtt_ingest_router import router as _ingest_router
        _ingest_router.reload_tree()
    except Exception:
        logger.warning('[asset-tree] ingest router reload failed', exc_info=True)

# ── Guards ────────────────────────────────────────────────────────────────
NAME_REGEX = re.compile(r'^[A-Za-z0-9_-]+$')
VALID_TOPIC_MODES = ('strict', 'learn')
VALID_DATA_TYPES = ('float', 'int', 'string')
# NOTE: no static MACHINE_LEVEL constant — see `_is_machine_level` which
# computes it from the live config so 5-level trees (e.g. datacenter →
# server) work identically to 4-level (factory → machine).


def _admin_only():
    """Return 403 JSON response if current user isn't admin, else None."""
    user = getattr(request, 'current_user', None)
    if not user or user.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    return None


def _sanitize_name(name: str):
    """Return (ok, err_msg). name must be topic-safe segment."""
    if not name or not isinstance(name, str):
        return False, 'name is required'
    if not NAME_REGEX.match(name):
        return False, "name must match ^[A-Za-z0-9_-]+$"
    if len(name) > 64:
        return False, 'name too long (max 64 chars)'
    return True, None


def _compute_topic_path(parent_id, name: str) -> str:
    """Join parent's topic_path with name using '/'. Root nodes = just name."""
    if parent_id is None:
        return name
    parent = AssetNode.get_by_id(parent_id)
    if not parent:
        return name
    return f"{parent['topic_path']}/{name}"


def _max_depth() -> int:
    """Max level index allowed by config's level_names length."""
    cfg = AssetTreeConfig.get()
    if not cfg or not cfg.get('level_names'):
        return 3  # default 4 levels → indices 0..3
    return max(0, len(cfg['level_names']) - 1)


# ── Config ─────────────────────────────────────────────────────────────────

@asset_tree_bp.route('/config', methods=['GET'])
@login_required
def get_config():
    """Frontend uses empty {} to know the setup wizard should run."""
    cfg = AssetTreeConfig.get()
    return jsonify(cfg or {})


@asset_tree_bp.route('/config', methods=['PUT'])
@login_required
def put_config():
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    level_names = data.get('level_names')
    root_name = data.get('root_name')
    topic_mode = data.get('topic_mode')
    meta_prefixes = data.get('meta_prefixes')

    # Validation
    if not isinstance(level_names, list) or not level_names:
        return jsonify({'error': 'level_names (non-empty list) required'}), 400
    if not all(isinstance(x, str) and x for x in level_names):
        return jsonify({'error': 'level_names must be non-empty strings'}), 400
    if not isinstance(root_name, str) or not root_name:
        return jsonify({'error': 'root_name (string) required'}), 400
    if topic_mode not in VALID_TOPIC_MODES:
        return jsonify({'error': f'topic_mode must be one of {VALID_TOPIC_MODES}'}), 400
    if not isinstance(meta_prefixes, list):
        return jsonify({'error': 'meta_prefixes (list) required'}), 400
    if not all(isinstance(x, str) for x in meta_prefixes):
        return jsonify({'error': 'meta_prefixes must be strings'}), 400

    before = AssetTreeConfig.get()
    after = AssetTreeConfig.upsert(level_names, root_name, topic_mode, meta_prefixes)
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='config_upsert',
        target_type='config',
        target_id=None,
        payload={'before': before, 'after': after},
    )
    _reload_ingest_router()
    return jsonify(after)


# ── Phase D — partial config PATCH ─────────────────────────────────────────
# Wizard uses PUT (full upsert); the Settings → MQTT Rules page uses PATCH
# so it can toggle `ingest_enabled` / adjust retention without having to
# resend level_names + root_name. Keeps the PUT contract stable for the
# wizard while unblocking Phase D UI.

@asset_tree_bp.route('/config', methods=['PATCH'])
@login_required
def patch_config():
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    updates = {}
    # Whitelist + basic type coercion. Silently drops unknown fields so a
    # forward-compat frontend can send extras without a 400.
    if 'topic_mode' in data:
        if data['topic_mode'] not in VALID_TOPIC_MODES:
            return jsonify({
                'error': f'topic_mode must be one of {VALID_TOPIC_MODES}'
            }), 400
        updates['topic_mode'] = data['topic_mode']
    if 'meta_prefixes' in data:
        mp = data['meta_prefixes']
        if not isinstance(mp, list) or not all(isinstance(x, str) for x in mp):
            return jsonify({'error': 'meta_prefixes must be a list of strings'}), 400
        # Trim + dedupe defensively. Users tend to paste trailing whitespace.
        updates['meta_prefixes'] = sorted({x.strip() for x in mp if x.strip()})
    if 'ingest_enabled' in data:
        updates['ingest_enabled'] = bool(data['ingest_enabled'])
    if 'ingest_retention_days' in data:
        try:
            rd = int(data['ingest_retention_days'])
        except (TypeError, ValueError):
            return jsonify({'error': 'ingest_retention_days must be an integer'}), 400
        if rd < 1 or rd > 3650:  # 10 years max
            return jsonify({'error': 'ingest_retention_days must be 1..3650'}), 400
        updates['ingest_retention_days'] = rd
    if 'level_names' in data or 'root_name' in data:
        # Structural changes must go through PUT so the wizard's validation
        # runs; blocking them here prevents surprise topic-path shifts.
        return jsonify({
            'error': 'level_names / root_name must be updated via PUT /config'
        }), 400

    if not updates:
        return jsonify({'error': 'No supported fields in body'}), 400

    before = AssetTreeConfig.get()
    if not before:
        return jsonify({
            'error': 'Config not found — run the setup wizard first'
        }), 404
    after = AssetTreeConfig.patch(**updates)
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='config_patch',
        target_type='config',
        target_id=None,
        payload={'before': before, 'after': after, 'updates': updates},
    )
    _reload_ingest_router()
    return jsonify(after)


# ── Presets ────────────────────────────────────────────────────────────────

@asset_tree_bp.route('/presets', methods=['GET'])
@login_required
def get_presets():
    return jsonify({
        'unit_presets': UNIT_PRESETS,
        'sample_rate_presets': SAMPLE_RATE_PRESETS,
        'sensor_templates': SENSOR_TEMPLATES,
        'hierarchy_presets': HIERARCHY_PRESETS,
    })


# ── Nodes ──────────────────────────────────────────────────────────────────

@asset_tree_bp.route('/nodes', methods=['GET'])
@login_required
def get_nodes():
    include_retired = request.args.get('include_retired', '').strip().lower() in ('1', 'true', 'yes')
    tree = AssetNode.tree_as_nested_json(include_retired=include_retired)
    return jsonify({'tree': tree})


@asset_tree_bp.route('/nodes', methods=['POST'])
@login_required
def create_node():
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    parent_id = data.get('parent_id')
    name = data.get('name')
    display_name = data.get('display_name')
    description = data.get('description')
    location_tag = data.get('location_tag')
    sensor_meta = data.get('sensor_meta')

    ok, err = _sanitize_name(name)
    if not ok:
        return jsonify({'error': err}), 400

    # Config must exist before node creation
    cfg = AssetTreeConfig.get()
    if not cfg:
        return jsonify({'error': 'Asset tree not yet configured; PUT /config first'}), 400

    max_depth = _max_depth()

    # Compute level from parent depth
    if parent_id is None:
        level = 0
        # Root name must match configured root_name (per config)
        # But we allow flexibility here: if a root doesn't yet exist, this
        # is fine. If one exists at this name AND level=0, we catch it via
        # the sibling-uniqueness check below.
    else:
        parent = AssetNode.get_by_id(parent_id)
        if not parent:
            return jsonify({'error': f'parent_id {parent_id} not found'}), 404
        if parent.get('status') == 'retired':
            return jsonify({'error': 'Cannot add child to a retired node'}), 400
        level = int(parent['level']) + 1
        if level > max_depth:
            return jsonify({
                'error': f'Parent already at max depth {max_depth}; cannot add child'
            }), 400

    # Sibling uniqueness (also enforced by UNIQUE(parent_id, name))
    if AssetNode.get_sibling_by_name(parent_id, name):
        return jsonify({'error': f"Sibling with name '{name}' already exists"}), 400

    topic_path = _compute_topic_path(parent_id, name)

    # topic_path must be unique globally (indexed UNIQUE)
    if AssetNode.get_by_topic_path(topic_path):
        return jsonify({'error': f"topic_path '{topic_path}' already in use"}), 400

    try:
        node_id = AssetNode.create(
            parent_id=parent_id,
            level=level,
            name=name,
            topic_path=topic_path,
            display_name=display_name,
            description=description,
            location_tag=location_tag,
            status='active',
        )
    except Exception as e:
        logger.exception('create_node failed')
        return jsonify({'error': str(e)}), 400

    # Attach sensor meta if provided (only meaningful at sensor / leaf level)
    if isinstance(sensor_meta, dict):
        dt = sensor_meta.get('data_type')
        if dt is not None and dt not in VALID_DATA_TYPES:
            return jsonify({'error': f'data_type must be one of {VALID_DATA_TYPES}'}), 400
        AssetSensorMeta.upsert(
            asset_id=node_id,
            unit=sensor_meta.get('unit'),
            sample_rate_hz=sensor_meta.get('sample_rate_hz'),
            expected_min=sensor_meta.get('expected_min'),
            expected_max=sensor_meta.get('expected_max'),
            data_type=dt,
        )

    created = AssetNode.get_by_id(node_id)
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='node_create',
        target_type='node',
        target_id=node_id,
        payload={'after': created, 'sensor_meta': sensor_meta},
    )
    _reload_ingest_router()
    return jsonify(created), 201


@asset_tree_bp.route('/nodes/<int:node_id>', methods=['PATCH'])
@login_required
def patch_node(node_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    node = AssetNode.get_by_id(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    # Refuse edits to retired nodes. If they need editing, an admin should
    # retire the tree and rebuild, OR reactivation semantics need to be
    # designed. Silently allowing PATCH creates confusing audit trails
    # (before=retired, after=active-with-new-name).
    if node.get('status') == 'retired':
        return jsonify({
            'error': 'Cannot patch a retired node. Retired is terminal for MVP.'
        }), 400
    data = request.get_json(silent=True) or {}

    updates = {}
    rename_from = None
    rename_to = None

    if 'name' in data:
        new_name = data['name']
        ok, err = _sanitize_name(new_name)
        if not ok:
            return jsonify({'error': err}), 400
        if new_name != node['name']:
            # Sibling collision check
            existing = AssetNode.get_sibling_by_name(node.get('parent_id'), new_name)
            if existing and existing['id'] != node_id:
                return jsonify({'error': f"Sibling with name '{new_name}' already exists"}), 400
            rename_from = node['name']
            rename_to = new_name
            updates['name'] = new_name

    for k in ('display_name', 'description', 'location_tag'):
        if k in data:
            updates[k] = data[k]

    sensor_meta = data.get('sensor_meta')

    if not updates and sensor_meta is None:
        return jsonify({'error': 'No valid fields to update'}), 400

    before = dict(node)

    # If renaming, recompute topic_path for this node AND all descendants
    if rename_to is not None:
        new_topic_path = _compute_topic_path(node.get('parent_id'), rename_to)
        # Guard against collisions on the new topic_path
        clash = AssetNode.get_by_topic_path(new_topic_path)
        if clash and clash['id'] != node_id:
            return jsonify({'error': f"topic_path '{new_topic_path}' already in use"}), 400
        updates['topic_path'] = new_topic_path

    if updates:
        AssetNode.update(node_id, **updates)

    # Recompute descendant topic_paths if we changed this node's topic_path
    if rename_to is not None:
        _recompute_subtree_topic_paths(node_id)

    if isinstance(sensor_meta, dict):
        dt = sensor_meta.get('data_type')
        if dt is not None and dt not in VALID_DATA_TYPES:
            return jsonify({'error': f'data_type must be one of {VALID_DATA_TYPES}'}), 400
        AssetSensorMeta.upsert(
            asset_id=node_id,
            unit=sensor_meta.get('unit'),
            sample_rate_hz=sensor_meta.get('sample_rate_hz'),
            expected_min=sensor_meta.get('expected_min'),
            expected_max=sensor_meta.get('expected_max'),
            data_type=dt,
        )

    after = AssetNode.get_by_id(node_id)
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='node_patch',
        target_type='node',
        target_id=node_id,
        payload={'before': before, 'after': after, 'sensor_meta': sensor_meta},
    )
    _reload_ingest_router()
    return jsonify(after)


def _recompute_subtree_topic_paths(root_id: int):
    """After a rename/move, recompute topic_paths for all descendants."""
    root = AssetNode.get_by_id(root_id)
    if not root:
        return
    root_path = root['topic_path']
    # BFS through descendants, recomputing each based on parent's new path
    with get_db() as conn:
        cursor = conn.cursor()
        queue = [(root_id, root_path)]
        while queue:
            parent_id, parent_path = queue.pop(0)
            cursor.execute(
                'SELECT id, name FROM asset_nodes WHERE parent_id = ?',
                (parent_id,)
            )
            for row in cursor.fetchall():
                child_id = row['id']
                child_name = row['name']
                new_path = f"{parent_path}/{child_name}"
                cursor.execute(
                    'UPDATE asset_nodes SET topic_path = ? WHERE id = ?',
                    (new_path, child_id)
                )
                queue.append((child_id, new_path))
        conn.commit()


@asset_tree_bp.route('/nodes/<int:node_id>/retire', methods=['POST'])
@login_required
def retire_node(node_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    node = AssetNode.get_by_id(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    if node.get('status') == 'retired':
        return jsonify({'error': 'Node already retired', 'node': node}), 400

    before = dict(node)
    affected = AssetNode.retire_cascade(node_id)
    after = AssetNode.get_by_id(node_id)

    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='node_retire',
        target_type='node',
        target_id=node_id,
        payload={'before': before, 'after': after, 'affected_ids': affected},
    )
    _reload_ingest_router()
    return jsonify({'retired_ids': affected, 'node': after})


@asset_tree_bp.route('/nodes/<int:node_id>/move', methods=['POST'])
@login_required
def move_node(node_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    node = AssetNode.get_by_id(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    data = request.get_json(silent=True) or {}
    new_parent_id = data.get('new_parent_id')

    if new_parent_id is None:
        return jsonify({'error': 'new_parent_id required'}), 400
    new_parent = AssetNode.get_by_id(new_parent_id)
    if not new_parent:
        return jsonify({'error': f'new_parent_id {new_parent_id} not found'}), 404

    # Current parent (may be None for root)
    current_parent_id = node.get('parent_id')
    if current_parent_id is None:
        return jsonify({'error': 'Cannot move a root node'}), 400
    current_parent = AssetNode.get_by_id(current_parent_id)

    # Same level check
    if current_parent and new_parent['level'] != current_parent['level']:
        return jsonify({
            'error': f"Parents must be at same level; current parent level={current_parent['level']}, "
                     f"new parent level={new_parent['level']}"
        }), 400

    # Prevent moving to descendant / self
    if new_parent_id == node_id:
        return jsonify({'error': 'Cannot move a node to itself'}), 400
    for d in AssetNode.get_descendants(node_id):
        if d['id'] == new_parent_id:
            return jsonify({'error': 'Cannot move a node to its own descendant'}), 400

    # Sibling uniqueness at new location
    clash = AssetNode.get_sibling_by_name(new_parent_id, node['name'])
    if clash and clash['id'] != node_id:
        return jsonify({
            'error': f"Sibling with name '{node['name']}' already exists at new parent"
        }), 400

    new_topic_path = _compute_topic_path(new_parent_id, node['name'])
    tp_clash = AssetNode.get_by_topic_path(new_topic_path)
    if tp_clash and tp_clash['id'] != node_id:
        return jsonify({'error': f"topic_path '{new_topic_path}' already in use"}), 400

    before = dict(node)
    AssetNode.update(node_id, parent_id=new_parent_id, topic_path=new_topic_path)
    _recompute_subtree_topic_paths(node_id)
    after = AssetNode.get_by_id(node_id)

    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='node_move',
        target_type='node',
        target_id=node_id,
        payload={'before': before, 'after': after,
                 'from_parent_id': current_parent_id,
                 'to_parent_id': new_parent_id},
    )
    _reload_ingest_router()
    return jsonify(after)


# ── Topic router ───────────────────────────────────────────────────────────

@asset_tree_bp.route('/topics/test', methods=['GET'])
@login_required
def test_topic():
    """Test a candidate MQTT topic against the tree.

    Returns:
      valid=true, route_to={node_id, topic_path}  → happy path (leaf match)
      valid=true, route_to=null, warnings=[...]    → meta topic (skipped)
      valid=false, reason='...', route_to=null     → invalid
    """
    topic = request.args.get('topic', '').strip()
    if not topic:
        return jsonify({'valid': False, 'reason': 'empty', 'route_to': None}), 200

    cfg = AssetTreeConfig.get()
    if not cfg:
        return jsonify({
            'valid': False,
            'reason': 'asset tree not configured',
            'route_to': None,
        }), 200

    segments = topic.split('/')
    root_name = cfg.get('root_name')
    meta_prefixes = cfg.get('meta_prefixes') or []

    # Root must match FIRST. Previously the meta-prefix short-circuit ran
    # before the root check, which let e.g. `wrong_root/_meta/x` slip through
    # as a "valid meta topic". Root check gates everything.
    if not segments or segments[0] != root_name:
        return jsonify({
            'valid': False,
            'reason': f"root mismatch — expected '{root_name}', got '{segments[0] if segments else ''}'",
            'route_to': None,
        }), 200

    # Meta prefix short-circuit: if ANY segment (after the root) is a meta
    # prefix, we accept the topic but don't route it to training data.
    # `seg == prefix` is a strict match — previously `seg.startswith(prefix)`
    # matched things like `_metastasized` because it starts with `_meta`.
    # If a delimiter-prefixed segment is truly needed (e.g. `_meta_v2`), it
    # must be added to meta_prefixes explicitly.
    meta_set = set(meta_prefixes)
    for seg in segments[1:]:
        if seg in meta_set:
            return jsonify({
                'valid': True,
                'route_to': None,
                'warnings': ['meta topic — not routed to training data'],
            }), 200

    # Walk the tree segment by segment, matching against topic_path
    accumulated = []
    current_parent_id = None
    matched_node = None
    for seg in segments:
        accumulated.append(seg)
        candidate_path = '/'.join(accumulated)
        node = AssetNode.get_by_topic_path(candidate_path)
        if not node:
            return jsonify({
                'valid': False,
                'reason': f"unknown segment '{seg}' at path '{candidate_path}'",
                'route_to': None,
            }), 200
        if node.get('status') == 'retired':
            return jsonify({
                'valid': False,
                'reason': f"segment '{seg}' is retired",
                'route_to': None,
            }), 200
        current_parent_id = node['id']
        matched_node = node

    return jsonify({
        'valid': True,
        'route_to': {
            'node_id': matched_node['id'],
            'topic_path': matched_node['topic_path'],
        },
        'warnings': [],
    }), 200


# ── Import ─────────────────────────────────────────────────────────────────

@asset_tree_bp.route('/import', methods=['POST'])
@login_required
def import_tree():
    """Bulk-build tree from a nested spec.

    Body: {'spec': {'name': 'factory', 'children': [{'name': 'plant_A', 'children': [...]}]}}
    Rejects if any node already exists.
    """
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    spec = data.get('spec')
    if not isinstance(spec, dict):
        return jsonify({'error': 'spec (dict) required'}), 400

    cfg = AssetTreeConfig.get()
    if not cfg:
        return jsonify({'error': 'PUT /config before importing'}), 400
    max_depth = _max_depth()

    # Import spec root MUST match the configured root_name — otherwise the
    # imported tree becomes unreachable by MQTT topics under strict mode.
    # Previously any root name was accepted, silently creating a second
    # top-level tree that contradicted the config.
    spec_root = spec.get('name')
    if spec_root != cfg.get('root_name'):
        return jsonify({
            'error': (
                f"spec root name '{spec_root}' does not match configured "
                f"root_name '{cfg.get('root_name')}'. Import must start "
                f"at the configured root."
            )
        }), 400

    created_ids = []

    def _walk(node_spec, parent_id, level, path_prefix):
        name = node_spec.get('name')
        ok, err = _sanitize_name(name)
        if not ok:
            raise ValueError(f'Bad name at level {level}: {err}')
        if level > max_depth:
            raise ValueError(f'Depth exceeds configured max_depth={max_depth}')
        topic_path = name if parent_id is None else f"{path_prefix}/{name}"
        if AssetNode.get_by_topic_path(topic_path):
            raise ValueError(f"Node '{topic_path}' already exists")
        node_id = AssetNode.create(
            parent_id=parent_id,
            level=level,
            name=name,
            topic_path=topic_path,
            display_name=node_spec.get('display_name'),
            description=node_spec.get('description'),
            location_tag=node_spec.get('location_tag'),
            status='active',
        )
        created_ids.append(node_id)
        sm = node_spec.get('sensor_meta')
        if isinstance(sm, dict):
            AssetSensorMeta.upsert(
                asset_id=node_id,
                unit=sm.get('unit'),
                sample_rate_hz=sm.get('sample_rate_hz'),
                expected_min=sm.get('expected_min'),
                expected_max=sm.get('expected_max'),
                data_type=sm.get('data_type'),
            )
        for child in node_spec.get('children') or []:
            _walk(child, node_id, level + 1, topic_path)

    try:
        _walk(spec, None, 0, '')
    except ValueError as e:
        # Best-effort rollback of anything already inserted
        with get_db() as conn:
            cur = conn.cursor()
            for cid in reversed(created_ids):
                try:
                    cur.execute('DELETE FROM asset_sensor_meta WHERE asset_id = ?', (cid,))
                    cur.execute('DELETE FROM asset_nodes WHERE id = ?', (cid,))
                except Exception:
                    pass
            conn.commit()
        return jsonify({'error': str(e)}), 400

    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='import',
        target_type='node',
        target_id=created_ids[0] if created_ids else None,
        payload={'created_ids': created_ids, 'spec': spec},
    )
    _reload_ingest_router()
    return jsonify({'created_ids': created_ids, 'count': len(created_ids)}), 201


# ── Tree templates ─────────────────────────────────────────────────────────
# Pre-built starter trees so users don't have to design a hierarchy from
# scratch. See constants/tree_templates.py for the catalog. Applying a
# template atomically sets config + imports the tree — used from the
# wizard Step 1 and from the admin empty-state.

@asset_tree_bp.route('/tree-templates', methods=['GET'])
@login_required
def list_tree_templates():
    """Return the full catalog of starter tree templates."""
    from ..constants.tree_templates import get_all
    return jsonify({'templates': get_all()})


@asset_tree_bp.route('/tree-templates/<template_id>/apply', methods=['POST'])
@login_required
def apply_tree_template(template_id):
    """Atomically seed config + tree from a starter template. Refuses if
    the tree is not empty — templates are for fresh installs only."""
    guard = _admin_only()
    if guard is not None:
        return guard
    from ..constants.tree_templates import get_by_id

    tmpl = get_by_id(template_id)
    if tmpl is None:
        return jsonify({'error': f'Unknown template: {template_id}'}), 404

    # Refuse if an ACTIVE tree already exists — we don't want to clobber
    # real work by accident. Retired roots are OK; they're kept for audit
    # but shouldn't block a fresh start. The admin "Reset & apply new
    # template" button retires the active root first, then calls this
    # endpoint. Any retired nodes we find get physically deleted here so
    # the new template's topic_paths don't collide against the UNIQUE
    # constraint — audit log entries in the separate audit table are
    # preserved either way.
    existing_roots = [r for r in AssetNode.get_children(None)
                      if r.get('status') != 'retired']
    if existing_roots:
        return jsonify({
            'error': (
                f'Active tree exists (root: {existing_roots[0]["name"]}). '
                f'Retire it first, then re-apply.'
            )
        }), 409
    # Physically delete any retired nodes to free up topic_paths.
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) AS n FROM asset_nodes')
        purged = cur.fetchone()['n']
        if purged:
            cur.execute('DELETE FROM asset_sensor_meta')
            cur.execute('DELETE FROM asset_nodes')
            conn.commit()
            AssetTreeAudit.log(
                actor_user_id=request.current_user['id'],
                event_type='tree_purge_retired',
                target_type='node',
                target_id=None,
                payload={'purged_count': purged,
                         'reason': 'apply_template requires clean slate'},
            )

    # Upsert the config bundled with the template.
    cfg = tmpl['config']
    AssetTreeConfig.upsert(
        level_names=cfg['level_names'],
        root_name=cfg['root_name'],
        topic_mode=cfg['topic_mode'],
        meta_prefixes=cfg['meta_prefixes'],
    )
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='config_upsert',
        target_type='config',
        target_id=None,
        payload={'source': 'template', 'template_id': template_id, 'config': cfg},
    )

    # Import the template's tree — re-use the import walker for consistency.
    max_depth = _max_depth()
    created_ids = []

    def _walk(node_spec, parent_id, level, path_prefix):
        name = node_spec.get('name')
        ok, err = _sanitize_name(name)
        if not ok:
            raise ValueError(f'Bad template name at level {level}: {err}')
        if level > max_depth:
            raise ValueError(f'Template exceeds configured max_depth={max_depth}')
        topic_path = name if parent_id is None else f"{path_prefix}/{name}"
        if AssetNode.get_by_topic_path(topic_path):
            raise ValueError(f"Node '{topic_path}' already exists (template collision)")
        node_id = AssetNode.create(
            parent_id=parent_id,
            level=level,
            name=name,
            topic_path=topic_path,
            display_name=node_spec.get('display_name'),
            description=node_spec.get('description'),
            location_tag=node_spec.get('location_tag'),
            status='active',
        )
        created_ids.append(node_id)
        sm = node_spec.get('sensor_meta')
        if isinstance(sm, dict):
            AssetSensorMeta.upsert(
                asset_id=node_id,
                unit=sm.get('unit'),
                sample_rate_hz=sm.get('sample_rate_hz'),
                expected_min=sm.get('expected_min'),
                expected_max=sm.get('expected_max'),
                data_type=sm.get('data_type'),
            )
        for child in node_spec.get('children') or []:
            _walk(child, node_id, level + 1, topic_path)

    try:
        _walk(tmpl['tree'], None, 0, '')
    except ValueError as e:
        # Best-effort rollback so the operator can retry cleanly.
        with get_db() as conn:
            cur = conn.cursor()
            for cid in reversed(created_ids):
                try:
                    cur.execute('DELETE FROM asset_sensor_meta WHERE asset_id = ?', (cid,))
                    cur.execute('DELETE FROM asset_nodes WHERE id = ?', (cid,))
                except Exception:
                    pass
            conn.commit()
        return jsonify({'error': str(e)}), 400

    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='template_apply',
        target_type='node',
        target_id=created_ids[0] if created_ids else None,
        payload={
            'template_id': template_id,
            'template_name': tmpl['name'],
            'created_ids': created_ids,
        },
    )
    _reload_ingest_router()
    return jsonify({
        'template_id': template_id,
        'created_ids': created_ids,
        'count': len(created_ids),
    }), 201


# ── Audit ──────────────────────────────────────────────────────────────────

@asset_tree_bp.route('/audit', methods=['GET'])
@login_required
def get_audit():
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'limit/offset must be integers'}), 400
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    rows = AssetTreeAudit.list(limit=limit, offset=offset)
    return jsonify({'audit': rows, 'total': AssetTreeAudit.count(),
                    'limit': limit, 'offset': offset})


# ── Groups ─────────────────────────────────────────────────────────────────

def _is_machine_level(asset_id: int) -> bool:
    node = AssetNode.get_by_id(asset_id)
    if not node:
        return False
    # If config's level_names has N entries: max_depth = N-1 (sensor).
    # Machine level = max_depth - 1 (parent of sensors). For default config
    # that's level 2. For 3-level configs it's level 1. This adapts.
    md = _max_depth()
    return int(node['level']) == max(0, md - 1)


def _is_active_machine(asset_id: int) -> tuple[bool, str | None]:
    """Return (True, None) if asset is an active machine; otherwise
    (False, reason). Used by group and rebind endpoints to prevent
    dead assets from creeping into scope lists. See Phase C QA #1/#6."""
    node = AssetNode.get_by_id(asset_id)
    if not node:
        return False, f'Asset id {asset_id} not found'
    md = _max_depth()
    if int(node['level']) != max(0, md - 1):
        return False, f'Asset id {asset_id} is not at machine level'
    if node.get('status') == 'retired':
        return False, f'Machine id {asset_id} ({node.get("name")}) is retired'
    return True, None


def _friendly_group_error(e: Exception, name: str) -> str:
    """Turn opaque SQL errors into copy the frontend can render. The
    UNIQUE-constraint case is the common one — we don't want the raw
    SQLite text leaking into user-facing tooltips (Phase C QA #4)."""
    msg = str(e)
    if 'UNIQUE constraint failed' in msg and 'machine_groups.name' in msg:
        return f"A group named '{name}' already exists."
    return msg


@asset_tree_bp.route('/groups', methods=['GET'])
@login_required
def list_groups():
    return jsonify({'groups': MachineGroup.get_all_with_counts()})


@asset_tree_bp.route('/groups', methods=['POST'])
@login_required
def create_group():
    guard = _admin_only()
    if guard is not None:
        return guard
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    description = data.get('description')
    machine_asset_ids = data.get('machine_asset_ids') or []

    if not name:
        return jsonify({'error': 'name is required'}), 400
    if not isinstance(machine_asset_ids, list):
        return jsonify({'error': 'machine_asset_ids must be a list'}), 400

    # All ids must be active machine-level nodes. Retired ones would
    # produce silently-broken groups (Phase C QA #1).
    for aid in machine_asset_ids:
        ok, err = _is_active_machine(aid)
        if not ok:
            return jsonify({'error': err}), 400

    try:
        gid = MachineGroup.create(
            name=name,
            description=description,
            created_by=request.current_user['id'],
        )
    except Exception as e:
        # Unique constraint on name is the most common — surface a friendly
        # message instead of the raw SQLite text (Phase C QA #4).
        return jsonify({'error': _friendly_group_error(e, name)}), 400
    MachineGroup.set_members(gid, machine_asset_ids)
    group = MachineGroup.get_by_id(gid)
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='group_create',
        target_type='group',
        target_id=gid,
        payload={'after': group, 'members': machine_asset_ids},
    )
    resp = dict(group)
    resp['members'] = MachineGroup.get_members(gid)
    return jsonify(resp), 201


@asset_tree_bp.route('/groups/<int:group_id>', methods=['GET'])
@login_required
def get_group(group_id):
    group = MachineGroup.get_by_id(group_id)
    if not group:
        return jsonify({'error': 'Group not found'}), 404
    group['members'] = MachineGroup.get_members(group_id)
    return jsonify(group)


@asset_tree_bp.route('/groups/<int:group_id>', methods=['PATCH'])
@login_required
def patch_group(group_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    group = MachineGroup.get_by_id(group_id)
    if not group:
        return jsonify({'error': 'Group not found'}), 404
    data = request.get_json(silent=True) or {}

    updates = {}
    if 'name' in data:
        new_name = (data['name'] or '').strip()
        if not new_name:
            return jsonify({'error': 'name cannot be empty'}), 400
        updates['name'] = new_name
    if 'description' in data:
        updates['description'] = data['description']

    before = dict(group)
    if updates:
        try:
            MachineGroup.update(group_id, **updates)
        except Exception as e:
            return jsonify({
                'error': _friendly_group_error(e, updates.get('name', ''))
            }), 400

    new_members = data.get('machine_asset_ids')
    if isinstance(new_members, list):
        # Reject retired machines here too — matches POST /groups semantics
        # (Phase C QA #1).
        for aid in new_members:
            ok, err = _is_active_machine(aid)
            if not ok:
                return jsonify({'error': err}), 400
        MachineGroup.set_members(group_id, new_members)

    after = MachineGroup.get_by_id(group_id)
    after['members'] = MachineGroup.get_members(group_id)

    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='group_patch',
        target_type='group',
        target_id=group_id,
        payload={'before': before, 'after': after},
    )
    return jsonify(after)


@asset_tree_bp.route('/groups/<int:group_id>', methods=['DELETE'])
@login_required
def delete_group(group_id):
    guard = _admin_only()
    if guard is not None:
        return guard
    group = MachineGroup.get_by_id(group_id)
    if not group:
        return jsonify({'error': 'Group not found'}), 404
    before = dict(group)
    before['members'] = MachineGroup.get_members(group_id)
    MachineGroup.delete(group_id)
    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='group_delete',
        target_type='group',
        target_id=group_id,
        payload={'before': before},
    )
    return jsonify({'message': 'Group deleted', 'id': group_id})


# ── Compatibility validator ────────────────────────────────────────────────

@asset_tree_bp.route('/validate-compatibility', methods=['POST'])
@login_required
def validate_compatibility():
    """Diff a set of machines against a reference (first in the list)."""
    data = request.get_json(silent=True) or {}
    machine_ids = data.get('machine_asset_ids') or []
    if not isinstance(machine_ids, list) or not machine_ids:
        return jsonify({'error': 'machine_asset_ids (non-empty list) required'}), 400

    machines = []
    for mid in machine_ids:
        node = AssetNode.get_by_id(mid)
        if not node:
            return jsonify({'error': f'Machine {mid} not found'}), 404
        if not _is_machine_level(mid):
            return jsonify({'error': f'Asset {mid} is not at machine level'}), 400
        machines.append(node)

    # Collect sensor children + their meta for each machine
    def _sensors_for(machine_node):
        sensors = AssetNode.get_children(machine_node['id'])
        # Filter to active only
        sensors = [s for s in sensors if s.get('status') == 'active']
        out = {}
        for s in sensors:
            meta = AssetSensorMeta.get(s['id']) or {}
            out[s['name']] = {
                'id': s['id'],
                'unit': meta.get('unit'),
                'sample_rate_hz': meta.get('sample_rate_hz'),
                'data_type': meta.get('data_type'),
            }
        return out

    ref_machine = machines[0]
    ref_sensors = _sensors_for(ref_machine)
    ref_names = set(ref_sensors.keys())

    per_machine_diff = []
    unit_mismatches = []
    sample_rate_mismatches = []
    compatible = True

    for m in machines[1:]:
        m_sensors = _sensors_for(m)
        m_names = set(m_sensors.keys())

        missing = sorted(list(ref_names - m_names))
        extra = sorted(list(m_names - ref_names))

        # Renamed sensors — best-effort heuristic: pair by matching unit +
        # sample_rate if both a "missing from m" and an "extra in m" share
        # the same unit + rate, treat as a rename.
        renamed = []
        remaining_missing = list(missing)
        remaining_extra = list(extra)
        for rn in list(remaining_missing):
            ref_meta = ref_sensors[rn]
            for en in list(remaining_extra):
                em = m_sensors[en]
                if (ref_meta.get('unit') == em.get('unit')
                        and ref_meta.get('sample_rate_hz') == em.get('sample_rate_hz')
                        and ref_meta.get('unit') is not None):
                    renamed.append({'from': rn, 'to': en})
                    remaining_missing.remove(rn)
                    remaining_extra.remove(en)
                    break

        # Overlapping names — check unit/rate mismatch
        overlap = ref_names & m_names
        for name in sorted(overlap):
            r = ref_sensors[name]
            mm = m_sensors[name]
            if r.get('unit') != mm.get('unit'):
                unit_mismatches.append({
                    'sensor': name,
                    'reference_machine_id': ref_machine['id'],
                    'other_machine_id': m['id'],
                    'reference_unit': r.get('unit'),
                    'other_unit': mm.get('unit'),
                })
            if r.get('sample_rate_hz') != mm.get('sample_rate_hz'):
                sample_rate_mismatches.append({
                    'sensor': name,
                    'reference_machine_id': ref_machine['id'],
                    'other_machine_id': m['id'],
                    'reference_sample_rate_hz': r.get('sample_rate_hz'),
                    'other_sample_rate_hz': mm.get('sample_rate_hz'),
                })

        if remaining_missing or remaining_extra or renamed:
            compatible = False

        per_machine_diff.append({
            'id': m['id'],
            'topic_path': m['topic_path'],
            'missing_sensors': remaining_missing,
            'extra_sensors': remaining_extra,
            'renamed_sensors': renamed,
        })

    if unit_mismatches or sample_rate_mismatches:
        compatible = False

    return jsonify({
        'compatible': compatible,
        'reference_machine': {
            'id': ref_machine['id'],
            'topic_path': ref_machine['topic_path'],
        },
        'per_machine_diff': per_machine_diff,
        'unit_mismatches': unit_mismatches,
        'sample_rate_mismatches': sample_rate_mismatches,
    })


# ── Model bindings (Phase B: Models + Deploy tabs) ─────────────────────────

@asset_tree_bp.route('/nodes/<int:node_id>/models', methods=['GET'])
@login_required
def get_machine_models(node_id):
    """Return the saved_models bound to this machine.

    Phase B — Models + Deploy tabs. Reads `model_machine_bindings`
    (Phase A migration) and joins to `saved_models` / `melab_endpoints`
    for display fields. Also returns "group models" — models the machine
    participates in via a machine_group.

    Response:
      {
        "trained_on":   [ { model, endpoints: [...] }, ... ],
        "deployed_to":  [ { model, endpoints: [...] }, ... ],
        "group_models": [ { model, endpoints: [...], group_id, group_name }, ... ]
      }

    A model may appear in both `trained_on` and `deployed_to` — same row,
    two roles. Empty lists on missing bindings; no 404 unless the node
    itself is absent.
    """
    node = AssetNode.get_by_id(node_id)
    if not node:
        return jsonify({'error': 'Node not found'}), 404
    if not _is_machine_level(node_id):
        return jsonify({
            'error': 'Node is not at machine level',
            'trained_on': [],
            'deployed_to': [],
            'group_models': [],
        }), 400

    def _model_row_with_endpoints(cursor, model_row):
        m = dict(model_row)
        cursor.execute(
            '''SELECT id, name, status, mode, algorithm, created_at
               FROM melab_endpoints WHERE saved_model_id = ?
               ORDER BY created_at DESC''',
            (m['id'],)
        )
        m['endpoints'] = [dict(r) for r in cursor.fetchall()]
        return m

    trained_on = []
    deployed_to = []
    group_models = []
    with get_db() as conn:
        cursor = conn.cursor()

        # Direct bindings
        cursor.execute(
            '''SELECT sm.*, mmb.role, mmb.trained_via_group, mmb.bound_at
               FROM model_machine_bindings mmb
               INNER JOIN saved_models sm ON sm.id = mmb.saved_model_id
               WHERE mmb.machine_asset_id = ?
               ORDER BY mmb.bound_at DESC''',
            (node_id,)
        )
        rows = cursor.fetchall()
        for r in rows:
            enriched = _model_row_with_endpoints(cursor, r)
            role = enriched.get('role')
            if role == 'deployed_to':
                deployed_to.append(enriched)
            else:
                # Treat any non-deployed role as "trained_on" — the current
                # writer only produces 'trained_on' / 'deployed_to'.
                trained_on.append(enriched)

        # Group models: any model bound to any machine in a group this
        # machine also participates in — excluding models already surfaced
        # under trained_on to avoid dupes.
        seen_ids = {m['id'] for m in trained_on} | {m['id'] for m in deployed_to}
        cursor.execute(
            '''SELECT DISTINCT g.id AS group_id, g.name AS group_name,
                              sm.id AS model_id
               FROM machine_group_members me_in
               INNER JOIN machine_groups g ON g.id = me_in.group_id
               INNER JOIN machine_group_members other ON other.group_id = g.id
               INNER JOIN model_machine_bindings mmb
                       ON mmb.machine_asset_id = other.machine_asset_id
               INNER JOIN saved_models sm ON sm.id = mmb.saved_model_id
               WHERE me_in.machine_asset_id = ?
                 AND other.machine_asset_id != ?''',
            (node_id, node_id)
        )
        rows = cursor.fetchall()
        for gr in rows:
            model_id = gr['model_id']
            if model_id in seen_ids:
                continue
            cursor.execute('SELECT * FROM saved_models WHERE id = ?', (model_id,))
            mrow = cursor.fetchone()
            if not mrow:
                continue
            enriched = _model_row_with_endpoints(cursor, mrow)
            enriched['group_id'] = gr['group_id']
            enriched['group_name'] = gr['group_name']
            group_models.append(enriched)
            seen_ids.add(model_id)

    return jsonify({
        'machine_id': node_id,
        'trained_on': trained_on,
        'deployed_to': deployed_to,
        'group_models': group_models,
    })


# ── Phase C: Model bindings (per-model view + rebind) ─────────────────────

@asset_tree_bp.route('/models/<int:saved_model_id>/bindings', methods=['GET'])
@login_required
def get_model_bindings(saved_model_id):
    """Return the machines this model was trained on and deployed to.

    Phase C.4. Read from `model_machine_bindings` (the join table shared with
    the machine-workspace Models tab, see /nodes/<id>/models above).

    Response:
      {
        "saved_model_id": 42,
        "trained_on":  [ { asset_id, name, topic_path, display_name,
                           asset_status }, ... ],
        "deployed_to": [ { asset_id, name, topic_path, display_name,
                           asset_status }, ... ],
        "trained_via_group": "All extruders" | null
      }
    """
    model = SavedModel.get_by_id(saved_model_id)
    if not model:
        return jsonify({'error': 'Saved model not found'}), 404

    rows = ModelMachineBinding.get_for_model(saved_model_id)

    def _project(r):
        return {
            'asset_id': r['machine_asset_id'],
            'name': r.get('asset_name'),
            'display_name': r.get('display_name'),
            'topic_path': r.get('topic_path'),
            'asset_status': r.get('asset_status'),
        }

    trained_on = [_project(r) for r in rows if r['role'] == 'trained_on']
    deployed_to = [_project(r) for r in rows if r['role'] == 'deployed_to']

    # Group snapshot lives on the 'trained_on' rows (writer sets it there).
    trained_via_group = None
    for r in rows:
        if r['role'] == 'trained_on' and r.get('trained_via_group'):
            trained_via_group = r['trained_via_group']
            break

    return jsonify({
        'saved_model_id': saved_model_id,
        'trained_on': trained_on,
        'deployed_to': deployed_to,
        'trained_via_group': trained_via_group,
    })


@asset_tree_bp.route('/models/<int:saved_model_id>/deploy-targets',
                     methods=['PATCH'])
@login_required
def patch_model_deploy_targets(saved_model_id):
    """Rebind a model's deploy targets. Admin only.

    Replaces every existing `role='deployed_to'` row for this model with the
    supplied list of machine asset ids. Trained-on bindings are untouched.

    Body: { "machine_asset_ids": [1, 5, 9] }
    """
    guard = _admin_only()
    if guard is not None:
        return guard

    model = SavedModel.get_by_id(saved_model_id)
    if not model:
        return jsonify({'error': 'Saved model not found'}), 404

    data = request.get_json(silent=True) or {}
    machine_ids = data.get('machine_asset_ids')
    if not isinstance(machine_ids, list):
        return jsonify({'error': 'machine_asset_ids must be a list'}), 400

    # Validate each id — must exist AND be at machine level. Retired machines
    # are permitted because rebinding to a retired machine is meaningful
    # (historical continuity). The training-scope path blocks retired.
    for mid in machine_ids:
        node = AssetNode.get_by_id(mid)
        if not node:
            return jsonify({'error': f'Asset id {mid} not found'}), 404
        if not _is_machine_level(mid):
            return jsonify({
                'error': f'Asset id {mid} is not at machine level'
            }), 400

    # Snapshot the pre-state for audit.
    before_rows = ModelMachineBinding.get_by_role(saved_model_id, 'deployed_to')

    ModelMachineBinding.replace_role(
        saved_model_id=saved_model_id,
        role='deployed_to',
        machine_asset_ids=machine_ids,
        # deployed_to rows don't need a trained_via_group snapshot — the
        # source of truth for group provenance is the trained_on rows.
        trained_via_group=None,
    )

    after_rows = ModelMachineBinding.get_by_role(saved_model_id, 'deployed_to')

    AssetTreeAudit.log(
        actor_user_id=request.current_user['id'],
        event_type='model_rebind',
        target_type='saved_model',
        target_id=saved_model_id,
        payload={
            'before_ids': [r['machine_asset_id'] for r in before_rows],
            'after_ids': [r['machine_asset_id'] for r in after_rows],
        },
    )

    return jsonify({
        'saved_model_id': saved_model_id,
        'deployed_to': [
            {
                'asset_id': r['machine_asset_id'],
                'name': r.get('asset_name'),
                'display_name': r.get('display_name'),
                'topic_path': r.get('topic_path'),
                'asset_status': r.get('asset_status'),
            } for r in after_rows
        ],
    })


# ── Phase C: Sample-count helper for scope selector ────────────────────────

@asset_tree_bp.route('/scope/sample-count', methods=['GET'])
@login_required
def scope_sample_count():
    """MVP helper for the Training-view scope selector.

    Accepts `?machine_ids=1,5,9`. Returns a rough dataset breakdown so the
    UI can render "N machines, M samples". Actual multi-machine pooling is
    deferred to Phase D — see ml_trainer.py TODO. Until then, we return the
    per-machine on-disk dataset count so the scope card can display SOMETHING
    for group / ad-hoc modes instead of an em-dash.

    Response:
      {
        "machine_ids": [1, 5, 9],
        "machines": [
          { "id": 1, "name": "server_1", "topic_path": "...",
            "sensor_count": 4 }
        ],
        "sample_count": null,     ← MVP; wire to on-disk scan in Phase D
        "class_count": null
      }
    """
    ids_param = request.args.get('machine_ids', '')
    try:
        machine_ids = [
            int(x) for x in ids_param.split(',') if x.strip()
        ]
    except ValueError:
        return jsonify({'error': 'machine_ids must be a comma-separated list of integers'}), 400

    machines = []
    for mid in machine_ids:
        node = AssetNode.get_by_id(mid)
        if not node:
            continue
        if not _is_machine_level(mid):
            continue
        children = AssetNode.get_children(mid) or []
        machines.append({
            'id': mid,
            'name': node.get('name'),
            'display_name': node.get('display_name'),
            'topic_path': node.get('topic_path'),
            'status': node.get('status'),
            'sensor_count': sum(1 for c in children if c.get('status') == 'active'),
        })

    return jsonify({
        'machine_ids': machine_ids,
        'machines': machines,
        # Sample / class counts require the multi-machine pooling that
        # Phase D ships. Frontend renders "—" when null.
        'sample_count': None,
        'class_count': None,
    })


# ── Phase D — MQTT ingest router endpoints ─────────────────────────────────
# All admin-gated for writes; reads open to any logged-in user (matches the
# rest of asset_tree). All are strictly additive — no existing endpoint's
# contract changes. See services/mqtt_ingest_router.py for the machinery.

@asset_tree_bp.route('/ingest-stats', methods=['GET'])
@login_required
def get_ingest_stats():
    """Return the router's in-memory counters + thread liveness. Open to
    any authenticated user because the Stats tab is a read-only debug view."""
    try:
        from ..services.mqtt_ingest_router import router as _ingest_router
        return jsonify(_ingest_router.snapshot())
    except Exception as e:
        logger.warning('[asset-tree] ingest-stats failed: %s', e)
        return jsonify({
            'error': 'ingest router unavailable',
            'enabled': False,
            'connected': False,
        }), 200


@asset_tree_bp.route('/rejected-topics', methods=['GET'])
@login_required
def get_rejected_topics():
    """List rejected topics for a given date (default: today UTC).

    Query params:
      date  — YYYY-MM-DD (defaults to today)
      limit — max entries (default 200, capped at 5000)
    """
    date = request.args.get('date', '').strip() or None
    try:
        limit = int(request.args.get('limit', 200))
    except (TypeError, ValueError):
        limit = 200
    try:
        from ..services.mqtt_ingest_router import router as _ingest_router
        entries = _ingest_router.list_rejected(date=date, limit=limit)
        return jsonify({'date': date, 'entries': entries, 'count': len(entries)})
    except Exception as e:
        logger.warning('[asset-tree] rejected-topics failed: %s', e)
        return jsonify({'date': date, 'entries': [], 'count': 0}), 200


@asset_tree_bp.route('/ingest-janitor/run-now', methods=['POST'])
@login_required
def run_ingest_janitor():
    """Force a synchronous retention sweep. Admin-only.

    Intended for QA — surfacing this in the UI is optional. The background
    janitor runs every 6 h regardless; this just gives operators a way to
    trigger it manually without waiting.
    """
    guard = _admin_only()
    if guard is not None:
        return guard
    try:
        from ..services.mqtt_ingest_router import router as _ingest_router
        summary = _ingest_router.run_janitor_once()
        AssetTreeAudit.log(
            actor_user_id=request.current_user['id'],
            event_type='ingest_janitor_run',
            target_type='config',
            target_id=None,
            payload=summary,
        )
        return jsonify(summary)
    except Exception as e:
        logger.exception('[asset-tree] janitor run-now failed')
        return jsonify({'error': str(e)}), 500
