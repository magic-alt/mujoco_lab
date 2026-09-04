from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mujoco_lab.robots.resolver import ResolvedRobotModel
from mujoco_lab.robots.spec import RobotModelSpec

G1_CONTROLLED_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

G1_SPEC = RobotModelSpec(
    name="unitree-g1-29dof",
    source_repository="google-deepmind/mujoco_menagerie",
    revision="e4049d0a3bfd58d2a3081614e6777d4007e3f86a",
    model_subdir="unitree_g1",
    model_file="g1.xml",
    scene_file="scene.xml",
    license_id="BSD-3-Clause",
    expected_joint_count=29,
    expected_actuator_count=29,
    required_sites=("imu_in_pelvis", "imu_in_torso", "left_foot", "right_foot"),
    required_bodies=("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"),
    required_controlled_joints=G1_CONTROLLED_JOINTS,
)


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        mujoco.mj_id2name(model, object_type, index) or f"<unnamed:{index}>"
        for index in range(count)
    ]


def _require_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"required MuJoCo object is missing: {object_type.name}/{name}")
    return object_id


def _stand_keyframe(model: mujoco.MjModel) -> tuple[int, np.ndarray]:
    key_id = _require_name(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    return key_id, np.asarray(model.key_qpos[key_id], dtype=np.float64).copy()


def _validate_nominal_joint_limits(model: mujoco.MjModel, stand_qpos: np.ndarray) -> None:
    violations: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        lower, upper = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        value = float(stand_qpos[qpos_address])
        if value < lower or value > upper:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append(f"{name}: {value:.6f} not in [{lower:.6f}, {upper:.6f}]")
    if violations:
        raise ValueError("stand keyframe violates joint limits: " + "; ".join(violations))


def _foot_contact_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = _require_name(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and int(model.geom_contype[geom_id]) != 0
    ]


def inspect_g1_model(resolved: ResolvedRobotModel) -> dict[str, Any]:
    """Compile and validate the pinned Menagerie G1 before task code can use it."""

    model = mujoco.MjModel.from_xml_path(str(resolved.scene_path))
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    articulated_joint_names = [
        name
        for joint_id, name in enumerate(joint_names)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    body_names = set(_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody))
    site_names = set(_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))

    if (
        G1_SPEC.expected_joint_count is not None
        and len(articulated_joint_names) != G1_SPEC.expected_joint_count
    ):
        raise ValueError(
            f"G1 articulated joint count changed: expected {G1_SPEC.expected_joint_count}, "
            f"found {len(articulated_joint_names)}"
        )
    if G1_SPEC.expected_actuator_count is not None and model.nu != G1_SPEC.expected_actuator_count:
        raise ValueError(
            f"G1 actuator count changed: expected {G1_SPEC.expected_actuator_count}, found {model.nu}"
        )

    missing_controlled = sorted(
        set(G1_SPEC.required_controlled_joints) - set(articulated_joint_names)
    )
    if missing_controlled:
        raise ValueError(f"G1 controlled joints are missing: {missing_controlled}")
    missing_actuators = sorted(set(articulated_joint_names) - set(actuator_names))
    if missing_actuators:
        raise ValueError(f"G1 joint-matched actuators are missing: {missing_actuators}")
    missing_bodies = sorted(set(G1_SPEC.required_bodies) - body_names)
    if missing_bodies:
        raise ValueError(f"G1 required bodies are missing: {missing_bodies}")
    missing_sites = sorted(set(G1_SPEC.required_sites) - site_names)
    if missing_sites:
        raise ValueError(f"G1 required sites are missing: {missing_sites}")

    key_id, stand_qpos = _stand_keyframe(model)
    _validate_nominal_joint_limits(model, stand_qpos)

    left_foot_geoms = _foot_contact_geom_ids(model, "left_ankle_roll_link")
    right_foot_geoms = _foot_contact_geom_ids(model, "right_ankle_roll_link")
    if len(left_foot_geoms) < 4 or len(right_foot_geoms) < 4:
        raise ValueError(
            "G1 foot contact contract changed: expected at least four collision geoms per foot, "
            f"found left={len(left_foot_geoms)} right={len(right_foot_geoms)}"
        )

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("G1 produced non-finite state during the one-step inspection rollout")

    actuators: list[dict[str, Any]] = []
    for actuator_id, actuator_name in enumerate(actuator_names):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuators.append(
            {
                "name": actuator_name,
                "joint": joint_name,
                "ctrl_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "ctrl_range": np.asarray(model.actuator_ctrlrange[actuator_id]).tolist(),
                "joint_force_limited": bool(model.jnt_actfrclimited[joint_id]),
                "joint_force_range": np.asarray(model.jnt_actfrcrange[joint_id]).tolist(),
            }
        )

    return {
        "robot": G1_SPEC.name,
        "source_repository": G1_SPEC.source_repository,
        "source_url": G1_SPEC.source_url,
        "revision": G1_SPEC.revision,
        "license_id": G1_SPEC.license_id,
        "license": str(resolved.license_path),
        "cache_root": str(resolved.root),
        "scene": str(resolved.scene_path),
        "nq": model.nq,
        "nv": model.nv,
        "articulated_joint_count": len(articulated_joint_names),
        "actuator_count": model.nu,
        "timestep": float(model.opt.timestep),
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "controlled_joints": list(G1_SPEC.required_controlled_joints),
        "required_sites": list(G1_SPEC.required_sites),
        "left_foot_contact_geom_ids": left_foot_geoms,
        "right_foot_contact_geom_ids": right_foot_geoms,
        "stand_base_height": float(stand_qpos[2]),
        "actuators": actuators,
    }


def inspection_json_path(resolved: ResolvedRobotModel) -> Path:
    """Return the conventional cache-local path for a serialized inspection report."""

    return resolved.root / "inspection.json"
