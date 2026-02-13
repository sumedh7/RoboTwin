"""Visualize reasoning point annotations overlaid on head-camera images.

Produces a grid of frames from selected episodes showing the pick (green)
and place (red) reasoning points on the head-camera image.

Usage (from policy/pi0/):
    uv run scripts/visualize_reasoning_points.py [--episodes 0 1 2] [--frames-per-ep 6]
"""

import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def decode_image(img_dict) -> np.ndarray:
    """Decode a LeRobot image dict (with 'bytes' key) to a numpy HWC uint8 BGR array."""
    raw = img_dict["bytes"]
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)  # RGB HWC
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_point(img, nx, ny, color, radius=10, thickness=2, label=None):
    """Draw a circle + optional label at normalised coords (nx, ny)."""
    h, w = img.shape[:2]
    px = int(nx * w)
    py = int(ny * h)
    cv2.circle(img, (px, py), radius, color, thickness)
    cv2.circle(img, (px, py), 2, color, -1)  # centre dot
    if label:
        cv2.putText(img, label, (px + radius + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str,
                        default=os.path.expanduser(
                            "~/.cache/huggingface/lerobot/demo_randomized_place_anyobject_stand_5k"))
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--frames-per-ep", type=int, default=8,
                        help="Number of evenly-spaced frames to show per episode")
    parser.add_argument("--out", type=str, default="reasoning_points_vis.png")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    data_dir = dataset_dir / "data"

    rows = []
    for ep_idx in args.episodes:
        chunk = ep_idx // 1000
        pf = data_dir / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not pf.exists():
            print(f"Episode {ep_idx} not found at {pf}, skipping")
            continue

        table = pq.read_table(pf)
        df = table.to_pandas()

        if "observation.reasoning_point" not in df.columns:
            print(f"Episode {ep_idx}: reasoning_point column not found, skipping")
            continue

        n_frames = len(df)
        indices = np.linspace(0, n_frames - 1, args.frames_per_ep, dtype=int)

        # Parse reasoning points and gripper state
        reasoning_pts = np.array([r for r in df["observation.reasoning_point"]])
        left_gripper = np.array([s[6] for s in df["observation.state"]])
        right_gripper = np.array([s[13] for s in df["observation.state"]])

        frame_imgs = []
        for fi in indices:
            img = decode_image(df["observation.images.cam_high"].iloc[fi])
            rx, ry = reasoning_pts[fi]

            # Determine phase: pick (before grasp) or place (after grasp)
            lg, rg = left_gripper[fi], right_gripper[fi]
            grasping = (lg < 0.3) or (rg < 0.3)

            if grasping:
                color = (0, 0, 255)  # Red = place phase
                label = "place"
            else:
                # Check if gripper has been closed at all up to this frame
                past_closed_l = np.any(left_gripper[:fi + 1] < 0.3)
                past_closed_r = np.any(right_gripper[:fi + 1] < 0.3)
                if past_closed_l or past_closed_r:
                    color = (0, 0, 255)  # Red = place phase (after grasp, gripper reopened)
                    label = "place"
                else:
                    color = (0, 200, 0)  # Green = pick phase
                    label = "pick"

            draw_point(img, rx, ry, color, label=label)

            # Add frame number
            cv2.putText(img, f"f{fi}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            frame_imgs.append(img)

        # Concatenate frames horizontally
        row = np.concatenate(frame_imgs, axis=1)

        # Add episode label on the left
        label_w = 80
        label_img = np.zeros((row.shape[0], label_w, 3), dtype=np.uint8)
        cv2.putText(label_img, f"Ep {ep_idx}", (5, row.shape[0] // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        row = np.concatenate([label_img, row], axis=1)
        rows.append(row)

    if not rows:
        print("No episodes to visualise!")
        return

    # Pad rows to same width if needed
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)

    grid = np.concatenate(padded, axis=0)
    cv2.imwrite(args.out, grid)
    print(f"Saved visualisation to {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
