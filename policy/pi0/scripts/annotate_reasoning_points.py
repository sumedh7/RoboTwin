"""Annotate the LeRobot dataset with pick/place reasoning points.

For each frame in each episode:
  - Before the grasp: the reasoning point is the PICK point (where the grasp will occur)
  - After the grasp: the reasoning point is the PLACE point (where the placement will occur)

The reasoning point is expressed in normalized head camera pixel space [0, 1].

Usage (from policy/pi0/):
    uv run scripts/annotate_reasoning_points.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# ===========================================================================
# Forward kinematics helpers
# ===========================================================================

def _rpy_to_matrix(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _make_transform(xyz, rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _revolute_transform(angle, axis):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(angle * np.array(axis, dtype=float)).as_matrix()
    return T


# Gripper-tip offset from link6 (midpoint between the two finger prismatic joints)
_GRIPPER_TIP_OFFSET = [0.08457, 0, 0]


def fk_left_arm(joint_angles):
    """6-DOF FK for the left (fl_) arm.  Returns 4x4 homogeneous transform in the robot frame."""
    T = _make_transform([0.2305, 0.297, 0.782], [0.0, 0.0, 0.02])
    T = T @ _make_transform([0, 0, 0.058])
    T = T @ _revolute_transform(joint_angles[0], [0, 0, 1])
    T = T @ _make_transform([0.025013, 0.00060169, 0.042])
    T = T @ _revolute_transform(joint_angles[1], [0, 1, 0])
    T = T @ _make_transform([-0.26396, 0.0044548, 0], [-3.1416, 0, -0.015928])
    T = T @ _revolute_transform(joint_angles[2], [0, 1, 0])
    T = T @ _make_transform([0.246, -0.00025, -0.06])
    T = T @ _revolute_transform(joint_angles[3], [0, 1, 0])
    T = T @ _make_transform([0.06775, 0.0015, -0.0855], [0, 0, -0.015928])
    T = T @ _revolute_transform(joint_angles[4], [0, 0, 1])
    T = T @ _make_transform([0.03095, 0, 0.0855], [-3.1416, 0, 0])
    T = T @ _revolute_transform(joint_angles[5], [1, 0, 0])
    T = T @ _make_transform(_GRIPPER_TIP_OFFSET)
    return T


def fk_right_arm(joint_angles):
    """6-DOF FK for the right (fr_) arm.  Returns 4x4 homogeneous transform in the robot frame."""
    T = _make_transform([0.2315, -0.3063, 0.781], [0.0, 0.0, 0.01])
    T = T @ _make_transform([0, 0, 0.058])
    T = T @ _revolute_transform(joint_angles[0], [0, 0, 1])
    T = T @ _make_transform([0.025013, 0.00060169, 0.042])
    T = T @ _revolute_transform(joint_angles[1], [0, 1, 0])
    T = T @ _make_transform([-0.26396, 0.0044548, 0], [-3.1416, 0, -0.015928])
    T = T @ _revolute_transform(joint_angles[2], [0, 1, 0])
    T = T @ _make_transform([0.246, -0.00025, -0.06])
    T = T @ _revolute_transform(joint_angles[3], [0, 1, 0])
    T = T @ _make_transform([0.06775, 0.0015, -0.0855], [0, 0, -0.015928])
    T = T @ _revolute_transform(joint_angles[4], [0, 0, 1])
    T = T @ _make_transform([0.03095, 0, 0.0855], [-3.1416, 0, 0])
    T = T @ _revolute_transform(joint_angles[5], [1, 0, 0])
    T = T @ _make_transform(_GRIPPER_TIP_OFFSET)
    return T


# ===========================================================================
# Camera projection helpers
# ===========================================================================

# Robot base pose in the world (from aloha-agilex config.yml)
_ROBOT_POS = np.array([0, -0.65, 0.0])
_ROBOT_QUAT_XYZW = [0, 0, 0.707, 0.707]  # 90° about z (scipy convention)
_ROBOT_T = np.eye(4)
_ROBOT_T[:3, :3] = Rotation.from_quat(_ROBOT_QUAT_XYZW).as_matrix()
_ROBOT_T[:3, 3] = _ROBOT_POS

# Head camera (from aloha-agilex config.yml)
_CAM_POS = np.array([-0.032, -0.45, 1.35])
_CAM_FORWARD = np.array([0, 0.6, -0.8])
_CAM_FORWARD = _CAM_FORWARD / np.linalg.norm(_CAM_FORWARD)
_CAM_LEFT = np.array([-1.0, 0, 0])
_CAM_UP = np.cross(_CAM_FORWARD, _CAM_LEFT)

_CAM2WORLD = np.eye(4)
_CAM2WORLD[:3, :3] = np.stack([_CAM_FORWARD, _CAM_LEFT, _CAM_UP], axis=1)
_CAM2WORLD[:3, 3] = _CAM_POS
_WORLD2CAM = np.linalg.inv(_CAM2WORLD)

# SAPIEN camera frame → OpenCV camera frame
# SAPIEN: x=forward, y=left, z=up
# OpenCV: x=right,   y=down, z=forward
_SAPIEN_TO_CV = np.array([
    [0, -1, 0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1],
], dtype=float)

# Intrinsics  (D435: fovy=37°, native 320×240, stored as 640×480)
_FOVY_DEG = 37.0
_NATIVE_H, _NATIVE_W = 240, 320
_STORED_H, _STORED_W = 480, 640
_SCALE = _STORED_H / _NATIVE_H

_FY_NATIVE = (_NATIVE_H / 2) / np.tan(np.deg2rad(_FOVY_DEG / 2))
_FX = _FY_NATIVE * _SCALE
_FY = _FY_NATIVE * _SCALE
_CX = _STORED_W / 2.0
_CY = _STORED_H / 2.0


def _project_to_normalized_pixel(world_point: np.ndarray) -> tuple[float, float]:
    """Project a 3-D world point to normalised (0-1) head-camera pixel coords."""
    p_cam = _WORLD2CAM @ np.append(world_point, 1.0)
    p_cv = _SAPIEN_TO_CV @ p_cam
    if p_cv[2] <= 0:
        return 0.5, 0.5  # behind camera – fall back to centre
    u = _FX * p_cv[0] / p_cv[2] + _CX
    v = _FY * p_cv[1] / p_cv[2] + _CY
    # Normalise & clamp to [0, 1]
    nx = float(np.clip(u / _STORED_W, 0, 1))
    ny = float(np.clip(v / _STORED_H, 0, 1))
    return nx, ny


# ===========================================================================
# Episode annotation logic
# ===========================================================================

_GRIPPER_CLOSE_THRESHOLD = 0.3   # gripper considered closed below this
_GRIPPER_OPEN_THRESHOLD = 0.5    # gripper considered open above this


def _find_grasp_release(gripper_traj: np.ndarray):
    """Return (grasp_frame, release_frame) indices.

    grasp_frame  : first frame where gripper falls below close threshold
    release_frame: first frame *after* grasp where gripper rises above open threshold
    """
    closed_mask = gripper_traj < _GRIPPER_CLOSE_THRESHOLD
    if not np.any(closed_mask):
        return None, None
    grasp_frame = int(np.argmax(closed_mask))

    opened_after = gripper_traj[grasp_frame:] > _GRIPPER_OPEN_THRESHOLD
    if not np.any(opened_after):
        release_frame = len(gripper_traj) - 1
    else:
        release_frame = grasp_frame + int(np.argmax(opened_after))

    return grasp_frame, release_frame


def annotate_episode(states: np.ndarray):
    """Compute (reasoning_x, reasoning_y) for every frame.

    Parameters
    ----------
    states : (T, 14)  joint states for the episode

    Returns
    -------
    reasoning_points : (T, 2)  normalised (x, y) pixel coordinates
    """
    T = states.shape[0]
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]

    # Determine which arm is active
    left_range = left_gripper.max() - left_gripper.min()
    right_range = right_gripper.max() - right_gripper.min()

    if left_range >= right_range:
        fk_fn = fk_left_arm
        joint_slice = slice(0, 6)
        gripper_traj = left_gripper
    else:
        fk_fn = fk_right_arm
        joint_slice = slice(7, 13)
        gripper_traj = right_gripper

    grasp_frame, release_frame = _find_grasp_release(gripper_traj)

    if grasp_frame is None:
        # No grasp detected – compute EE position at the middle frame as fallback
        mid = T // 2
        joints_mid = states[mid, joint_slice]
        T_mid = _ROBOT_T @ fk_fn(joints_mid)
        px, py = _project_to_normalized_pixel(T_mid[:3, 3])
        return np.full((T, 2), [px, py], dtype=np.float32)

    # Pick point: EE at the grasp frame
    joints_grasp = states[grasp_frame, joint_slice]
    T_grasp = _ROBOT_T @ fk_fn(joints_grasp)
    pick_x, pick_y = _project_to_normalized_pixel(T_grasp[:3, 3])

    # Place point: EE at the release frame
    joints_release = states[release_frame, joint_slice]
    T_release = _ROBOT_T @ fk_fn(joints_release)
    place_x, place_y = _project_to_normalized_pixel(T_release[:3, 3])

    reasoning = np.empty((T, 2), dtype=np.float32)
    reasoning[:grasp_frame] = [pick_x, pick_y]
    reasoning[grasp_frame:] = [place_x, place_y]

    return reasoning


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Annotate reasoning points")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.expanduser(
            "~/.cache/huggingface/lerobot/demo_randomized_place_anyobject_stand_5k"
        ),
        help="Root of the LeRobot dataset",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print, don't write")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    data_dir = dataset_dir / "data"
    meta_dir = dataset_dir / "meta"

    # Discover all episode parquet files
    parquet_files = sorted(data_dir.rglob("episode_*.parquet"))
    print(f"Found {len(parquet_files)} episode files")

    all_reasoning = []  # collect for stats
    for i, pf in enumerate(parquet_files):
        # Only read the state column first (avoid loading huge image blobs)
        state_table = pq.read_table(pf, columns=["observation.state"])
        state_df = state_table.to_pandas()
        states = np.array([s for s in state_df["observation.state"]])

        reasoning = annotate_episode(states)
        all_reasoning.append(reasoning)

        if not args.dry_run:
            # Read the full table only when writing
            full_table = pq.read_table(pf)
            # Remove old column if it already exists (re-run safety)
            if "observation.reasoning_point" in full_table.column_names:
                idx = full_table.column_names.index("observation.reasoning_point")
                full_table = full_table.remove_column(idx)
            new_table = full_table.append_column(
                "observation.reasoning_point",
                pa.array(reasoning.tolist(), type=pa.list_(pa.float32())),
            )
            pq.write_table(new_table, pf)
            del full_table, new_table

        del state_table, state_df

        if (i + 1) % 500 == 0 or i == 0 or i == len(parquet_files) - 1:
            print(
                f"  [{i+1}/{len(parquet_files)}] {pf.name}  "
                f"frames={len(states)}  "
                f"pick=({reasoning[0, 0]:.3f},{reasoning[0, 1]:.3f})  "
                f"place=({reasoning[-1, 0]:.3f},{reasoning[-1, 1]:.3f})"
            )

    # Update info.json with new feature
    info_path = meta_dir / "info.json"
    if info_path.exists() and not args.dry_run:
        with open(info_path) as f:
            info = json.load(f)

        info["features"]["observation.reasoning_point"] = {
            "dtype": "float32",
            "shape": [2],
            "names": ["x", "y"],
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nUpdated {info_path}")

    # Print stats
    all_pts = np.concatenate(all_reasoning, axis=0)
    print(f"\n--- Reasoning point stats ({len(all_pts)} frames) ---")
    print(f"  x: min={all_pts[:, 0].min():.4f}  max={all_pts[:, 0].max():.4f}  mean={all_pts[:, 0].mean():.4f}")
    print(f"  y: min={all_pts[:, 1].min():.4f}  max={all_pts[:, 1].max():.4f}  mean={all_pts[:, 1].mean():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
