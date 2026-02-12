import sys

sys.path.append("./policy/ACT/")

import gc
import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_hdf5_actions_only(dataset_path):
    """Load only action/joint data (small). Caller must open file for image read."""
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
    return left_gripper, left_arm, right_gripper, right_arm


def read_frame_images(root, cam_names, j):
    """Read a single frame from each camera (avoids loading full episode into memory)."""
    return {
        cam_name: root[f"/observation/{cam_name}/rgb"][j]
        for cam_name in cam_names
    }


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def process_single_episode(i, path, save_path, cam_keys):
    """Process a single episode: read source HDF5, transform, and write output HDF5."""
    episode_path = os.path.join(path, f"episode{i}.hdf5")
    if not os.path.isfile(episode_path):
        print(f"Dataset does not exist at \n{episode_path}\n")
        exit()

    # ---- READ PHASE: single file open, bulk-read everything ----------------
    # Reading entire datasets at once ([:] / [()]) issues a few large
    # sequential I/O ops instead of num_steps * 3 tiny random reads.
    # This is the critical difference: HDF5 random indexing ([j]) goes through
    # the full HDF5 I/O path per call, which becomes a bottleneck with many
    # concurrent workers competing for disk bandwidth.
    with h5py.File(episode_path, "r") as root:
        left_gripper_all = root["/joint_action/left_gripper"][()]
        left_arm_all = root["/joint_action/left_arm"][()]
        right_gripper_all = root["/joint_action/right_gripper"][()]
        right_arm_all = root["/joint_action/right_arm"][()]
        # Bulk-read all frames per camera in one shot
        head_imgs_all = root["/observation/head_camera/rgb"][()]
        right_imgs_all = root["/observation/right_camera/rgb"][()]
        left_imgs_all = root["/observation/left_camera/rgb"][()]
    # HDF5 file is now closed — no file descriptor held during processing.

    num_steps = left_gripper_all.shape[0]
    n = num_steps - 1  # number of output frames
    state_dim = left_arm_all.shape[1] + 1 + right_arm_all.shape[1] + 1

    # Pre-allocate contiguous output arrays (avoids list resizing + np.stack copy)
    qpos = np.empty((n, state_dim), dtype=np.float32)
    actions = np.empty((n, state_dim), dtype=np.float32)
    cam_high = np.empty((n, 480, 640, 3), dtype=np.uint8)
    cam_right_wrist = np.empty((n, 480, 640, 3), dtype=np.uint8)
    cam_left_wrist = np.empty((n, 480, 640, 3), dtype=np.uint8)
    left_arm_dim = np.empty(n, dtype=np.int64)
    right_arm_dim = np.empty(n, dtype=np.int64)

    # ---- TRANSFORM PHASE: pure CPU work, no I/O ----------------------------
    state = None
    qi = 0  # index into qpos / image arrays
    ai = 0  # index into actions / arm_dim arrays

    for j in range(num_steps):
        left_gripper = left_gripper_all[j]
        left_arm = left_arm_all[j]
        right_gripper = right_gripper_all[j]
        right_arm = right_arm_all[j]

        if j != num_steps - 1:
            state = np.concatenate(
                (left_arm, [left_gripper], right_arm, [right_gripper]), axis=0
            ).astype(np.float32)
            qpos[qi] = state

            cam_high[qi] = cv2.resize(
                cv2.imdecode(np.frombuffer(head_imgs_all[j], np.uint8), cv2.IMREAD_COLOR),
                (640, 480),
            )
            cam_right_wrist[qi] = cv2.resize(
                cv2.imdecode(np.frombuffer(right_imgs_all[j], np.uint8), cv2.IMREAD_COLOR),
                (640, 480),
            )
            cam_left_wrist[qi] = cv2.resize(
                cv2.imdecode(np.frombuffer(left_imgs_all[j], np.uint8), cv2.IMREAD_COLOR),
                (640, 480),
            )
            qi += 1

        if j != 0:
            actions[ai] = state
            left_arm_dim[ai] = left_arm.shape[0]
            right_arm_dim[ai] = right_arm.shape[0]
            ai += 1

    # Free source data before writing
    del head_imgs_all, right_imgs_all, left_imgs_all
    del left_gripper_all, left_arm_all, right_gripper_all, right_arm_all

    # ---- WRITE PHASE -------------------------------------------------------
    # Image datasets are stored with gzip compression and per-frame chunking.
    # Raw uint8 images compress ~10-20x, cutting total write volume from ~2TB
    # to ~100-200GB for 5000 episodes.  This prevents the OS dirty-page cache
    # from saturating and blocking all writes (the root cause of the slowdown).
    # Per-frame chunks (1, H, W, 3) ensure efficient single-frame random access
    # during training, since h5py only decompresses the one chunk needed.
    IMG_CHUNKS = (1, 480, 640, 3)
    IMG_COMPRESS = "gzip"
    IMG_COMPRESS_LEVEL = 4

    hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")
    with h5py.File(hdf5path, "w") as f:
        f.create_dataset("action", data=actions)
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("left_arm_dim", data=left_arm_dim)
        obs.create_dataset("right_arm_dim", data=right_arm_dim)
        image = obs.create_group("images")
        image.create_dataset("cam_high", data=cam_high,
                             chunks=IMG_CHUNKS, compression=IMG_COMPRESS, compression_opts=IMG_COMPRESS_LEVEL)
        image.create_dataset("cam_right_wrist", data=cam_right_wrist,
                             chunks=IMG_CHUNKS, compression=IMG_COMPRESS, compression_opts=IMG_COMPRESS_LEVEL)
        image.create_dataset("cam_left_wrist", data=cam_left_wrist,
                             chunks=IMG_CHUNKS, compression=IMG_COMPRESS, compression_opts=IMG_COMPRESS_LEVEL)

    del qpos, actions, cam_high, cam_right_wrist, cam_left_wrist, left_arm_dim, right_arm_dim
    gc.collect()

    return i


def data_transform(path, episode_num, save_path, num_workers=1):
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Map from our camera key names to dataset keys (if different)
    cam_keys = ["head_camera", "right_camera", "left_camera"]

    worker_fn = partial(process_single_episode, path=path, save_path=save_path, cam_keys=cam_keys)

    if num_workers <= 1:
        # Sequential mode (original behaviour)
        for i in tqdm(range(episode_num), desc="Episodes", unit="ep"):
            worker_fn(i)
    else:
        # Parallel mode
        effective_workers = min(num_workers, episode_num)
        # maxtasksperchild: restart each worker after N episodes so that any
        # memory fragmentation from numpy/cv2/h5py internals is reclaimed by
        # the OS.  Keeps throughput stable across thousands of episodes.
        with Pool(processes=effective_workers, maxtasksperchild=50) as pool:
            for _ in tqdm(
                pool.imap_unordered(worker_fn, range(episode_num)),
                total=episode_num,
                desc=f"Episodes ({effective_workers} workers)",
                unit="ep",
            ):
                pass

    return episode_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for episode processing (default: 1, sequential)",
    )

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num
    num_workers = args.num_workers

    begin = 0
    begin = data_transform(
        os.path.join("../../data/", task_name, task_config, 'data'),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        num_workers=num_workers,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
