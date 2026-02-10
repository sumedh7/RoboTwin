#!/usr/bin/env python3
"""Merge parallel data-collection shards into the final data directory.

Each shard produces its own seed.txt, _traj_data/, data/, video/, and
scene_info.json under a separate directory.  This script re-indexes
everything into a single contiguous sequence of episodes.
"""

import os
import sys
import json
import shutil
import yaml
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Merge parallel collection shards")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("num_shards", type=int)
    parser.add_argument("--shard-base", type=str, default=None,
                        help="Root of shard directories (default: ./data/.shards/<task>/<config>)")
    parser.add_argument("--output", type=str, default=None,
                        help="Final output directory (default: read from config YAML)")
    args = parser.parse_args()

    # Resolve output path
    config_path = f"./task_config/{args.task_config}.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.output:
        out_dir = args.output
    else:
        base = config.get("save_path", "./data")
        out_dir = os.path.join(base, args.task_name, args.task_config)

    shard_base = args.shard_base or os.path.join(
        "./data/.shards", args.task_name, args.task_config)

    # Create destination dirs
    for sub in ("data", "video", "_traj_data"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    all_seeds = []
    merged_scene_info = {}
    episode_offset = 0

    for shard_id in range(args.num_shards):
        shard_dir = os.path.join(shard_base, f"shard_{shard_id}")
        if not os.path.isdir(shard_dir):
            continue

        # --- seeds ---
        seed_file = os.path.join(shard_dir, "seed.txt")
        shard_seeds = []
        if os.path.exists(seed_file):
            with open(seed_file) as f:
                shard_seeds = [int(s) for s in f.read().split() if s]
        num_eps = len(shard_seeds)
        all_seeds.extend(shard_seeds)

        # --- per-episode files ---
        moves = [
            ("_traj_data", "episode{}.pkl"),
            ("data",       "episode{}.hdf5"),
            ("video",      "episode{}.mp4"),
        ]
        for subdir, pattern in moves:
            src_dir = os.path.join(shard_dir, subdir)
            if not os.path.isdir(src_dir):
                continue
            for i in range(num_eps):
                src = os.path.join(src_dir, pattern.format(i))
                dst = os.path.join(out_dir, subdir, pattern.format(episode_offset + i))
                if os.path.exists(src):
                    shutil.move(src, dst)

        # --- scene_info.json ---
        info_path = os.path.join(shard_dir, "scene_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                shard_info = json.load(f)
            for key, value in shard_info.items():
                old_idx = int(key.split("_")[1])
                merged_scene_info[f"episode_{episode_offset + old_idx}"] = value

        print(f"  Shard {shard_id}: {num_eps} episodes  (running total: {episode_offset + num_eps})")
        episode_offset += num_eps

    # Write merged outputs
    with open(os.path.join(out_dir, "seed.txt"), "w") as f:
        f.write(" ".join(str(s) for s in all_seeds))

    with open(os.path.join(out_dir, "scene_info.json"), "w") as f:
        json.dump(merged_scene_info, f, ensure_ascii=False, indent=4)

    print(f"\n  Merged {episode_offset} episodes into {out_dir}")


if __name__ == "__main__":
    main()
