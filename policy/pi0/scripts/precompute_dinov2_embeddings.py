"""Precompute DINOv2 embeddings of future camera images for the LeRobot dataset.

For each frame *t* in each episode, this script computes the DINOv2 CLS-token
embedding of all three camera images at the *future* frame

    t_future = min(t + action_horizon - 1, last_frame_in_episode)

and stores the concatenated embedding as ``observation.future_dinov2_embedding``
(shape ``[num_cameras * dinov2_dim]``, e.g. ``[2304]`` for three cameras with
DINOv2 ViT-B/14).

This future embedding serves as a training target: the policy learns to predict
what the cameras *will* see at the end of the action chunk it is about to
produce.

All available GPUs are used in parallel (one worker per GPU).

Usage (from ``policy/pi0/``):

    .venv/bin/python scripts/precompute_dinov2_embeddings.py \\
        --dataset-dir /path/to/lerobot_dataset

Optional flags:

    --action-horizon 50          (default: 50)
    --dinov2-model dinov2_vitb14 (default: dinov2_vitb14)
    --batch-size 64              (default: 64)
    --dry-run                    (print stats only, don't write)
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

NUM_DECODE_WORKERS = 8  # threads for parallel JPEG decode + resize

# ── Defaults ──────────────────────────────────────────────────────────────────

CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# ImageNet normalisation constants used by DINOv2
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

COLUMN_NAME = "observation.future_dinov2_embedding"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dinov2(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()
    return model


def _get_raw_bytes(entry: dict, dataset_dir: Path) -> bytes:
    """Extract raw JPEG bytes from a parquet image entry."""
    raw = entry["bytes"]
    if raw is None:
        img_path = dataset_dir / entry["path"]
        with open(img_path, "rb") as f:
            raw = f.read()
    return raw


def _decode_image(raw_bytes: bytes) -> torch.Tensor:
    """Decode image bytes → [3, 224, 224] float tensor in [0, 1]."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return t


def _decode_entry(args: tuple[dict, Path]) -> torch.Tensor:
    """Decode a single parquet image entry (for use with ThreadPoolExecutor)."""
    entry, dataset_dir = args
    return _decode_image(_get_raw_bytes(entry, dataset_dir))


def _normalise_batch(batch: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalisation to a [B, 3, H, W] batch."""
    mean = _MEAN.to(batch.device)
    std = _STD.to(batch.device)
    return (batch - mean) / std


@torch.no_grad()
def _embed_images(model: torch.nn.Module, images: list[torch.Tensor], batch_size: int) -> np.ndarray:
    """Run DINOv2 on a list of [3, 224, 224] tensors; return [N, D] numpy."""
    device = next(model.parameters()).device
    all_feats = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start : start + batch_size]).to(device)
        batch = _normalise_batch(batch)
        feats = model(batch)  # [B, D]
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()


def _process_file(
    pf: Path,
    dataset_dir: Path,
    model: torch.nn.Module,
    action_horizon: int,
    batch_size: int,
    dry_run: bool,
    pool: ThreadPoolExecutor,
) -> int:
    """Process a single parquet file. Returns number of frames processed."""
    table = pq.read_table(pf)
    num_frames = table.num_rows

    # 1. Decode ALL camera images in parallel across all cameras at once.
    #    Flatten (cam, frame) pairs → thread pool → collect per-camera results.
    decode_args: list[tuple[dict, Path]] = []
    cam_lengths: list[int] = []
    for cam_key in CAMERA_KEYS:
        cam_col = table.column(cam_key)
        for row_idx in range(num_frames):
            decode_args.append((cam_col[row_idx].as_py(), dataset_dir))
        cam_lengths.append(num_frames)

    all_decoded = list(pool.map(_decode_entry, decode_args))

    # Split back into per-camera lists and embed
    offset = 0
    per_cam_embeddings = {}
    for cam_key, length in zip(CAMERA_KEYS, cam_lengths):
        images = all_decoded[offset : offset + length]
        offset += length
        per_cam_embeddings[cam_key] = _embed_images(model, images, batch_size)

    # 2. Concatenate cameras
    all_embeddings = np.concatenate(
        [per_cam_embeddings[k] for k in CAMERA_KEYS], axis=-1
    )

    # 3. Build future embeddings (look-ahead by action_horizon-1)
    future_indices = np.minimum(
        np.arange(num_frames) + action_horizon - 1,
        num_frames - 1,
    )
    future_embeddings = all_embeddings[future_indices]

    # 4. Write to parquet
    if not dry_run:
        if COLUMN_NAME in table.column_names:
            idx = table.column_names.index(COLUMN_NAME)
            table = table.remove_column(idx)
        new_table = table.append_column(
            COLUMN_NAME,
            pa.array(future_embeddings.tolist(), type=pa.list_(pa.float32())),
        )
        pq.write_table(new_table, pf)

    return num_frames


def _worker(
    gpu_id: int,
    parquet_files: list[Path],
    dataset_dir: Path,
    action_horizon: int,
    dinov2_model_name: str,
    batch_size: int,
    dry_run: bool,
    counter: mp.Value,
):
    """Worker function: runs on a single GPU, processes its shard of files."""
    device = torch.device(f"cuda:{gpu_id}")
    model = _load_dinov2(dinov2_model_name, device)

    with ThreadPoolExecutor(max_workers=NUM_DECODE_WORKERS) as pool:
        for pf in parquet_files:
            _process_file(pf, dataset_dir, model, action_horizon, batch_size, dry_run, pool)
            with counter.get_lock():
                counter.value += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute DINOv2 future-image embeddings")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root of the LeRobot dataset",
    )
    parser.add_argument("--action-horizon", type=int, default=50, help="Action chunk length")
    parser.add_argument("--dinov2-model", type=str, default="dinov2_vitb14", help="DINOv2 variant")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU batch size for DINOv2")
    parser.add_argument("--dry-run", action="store_true", help="Only print stats, don't write")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    data_dir = dataset_dir / "data"
    meta_dir = dataset_dir / "meta"

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, falling back to CPU (single-process)")
        num_gpus = 1

    # Discover episode parquet files
    parquet_files = sorted(data_dir.rglob("episode_*.parquet"))
    total_files = len(parquet_files)
    print(f"Found {total_files} episode files, using {num_gpus} GPU(s)\n")

    # Determine embedding dim from a quick forward pass on GPU 0
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp_model = _load_dinov2(args.dinov2_model, device0)
    dummy = torch.randn(1, 3, 224, 224, device=device0)
    dinov2_dim = tmp_model(dummy).shape[-1]
    embedding_dim = dinov2_dim * len(CAMERA_KEYS)
    del tmp_model, dummy
    torch.cuda.empty_cache()
    print(f"DINOv2 dim = {dinov2_dim}, cameras = {len(CAMERA_KEYS)}, total embedding dim = {embedding_dim}\n")

    if num_gpus == 1:
        # Single GPU / CPU — run in-process (no spawn overhead)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = _load_dinov2(args.dinov2_model, device)
        total_frames = 0
        pbar = tqdm(parquet_files, desc="Episodes", unit="ep", dynamic_ncols=True)
        with ThreadPoolExecutor(max_workers=NUM_DECODE_WORKERS) as pool:
            for pf in pbar:
                frames = _process_file(pf, dataset_dir, model, args.action_horizon, args.batch_size, args.dry_run, pool)
                total_frames += frames
                pbar.set_postfix(frames=frames, total_frames=total_frames)
    else:
        # Multi-GPU — split files round-robin across GPUs and spawn workers
        shards: list[list[Path]] = [[] for _ in range(num_gpus)]
        for i, pf in enumerate(parquet_files):
            shards[i % num_gpus].append(pf)

        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)

        processes = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=_worker,
                args=(
                    gpu_id, shards[gpu_id], dataset_dir, args.action_horizon,
                    args.dinov2_model, args.batch_size, args.dry_run,
                    counter,
                ),
            )
            p.start()
            processes.append(p)

        # Poll shared counter from main process to drive the progress bar
        pbar = tqdm(total=total_files, desc=f"Episodes ({num_gpus} GPUs)", unit="ep", dynamic_ncols=True)
        while any(p.is_alive() for p in processes):
            with counter.get_lock():
                done = counter.value
            pbar.update(done - pbar.n)
            time.sleep(0.5)
        # Final update
        with counter.get_lock():
            done = counter.value
        pbar.update(done - pbar.n)
        pbar.close()

        for p in processes:
            p.join()

        total_frames = sum(
            pq.read_metadata(pf).num_rows for pf in parquet_files
        )

    # ── Update info.json ──────────────────────────────────────────────────
    info_path = meta_dir / "info.json"
    if info_path.exists() and not args.dry_run:
        with open(info_path) as f:
            info = json.load(f)

        info["features"][COLUMN_NAME] = {
            "dtype": "float32",
            "shape": [embedding_dim],
            "names": None,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nUpdated {info_path}")

    print(f"\nDone! Processed {total_frames} frames across {total_files} episodes.")
    print(f"Embedding column: '{COLUMN_NAME}' (dim={embedding_dim})")


if __name__ == "__main__":
    main()
