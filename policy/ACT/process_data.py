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


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Map from our camera key names to dataset keys (if different)
    cam_keys = ["head_camera", "right_camera", "left_camera"]

    for i in tqdm(range(episode_num), desc="Episodes", unit="ep"):
        episode_path = os.path.join(path, f"episode{i}.hdf5")
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all = load_hdf5_actions_only(episode_path)

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        num_steps = left_gripper_all.shape[0]
        state = None

        # Open source file once and read images one frame at a time to limit memory
        with h5py.File(episode_path, "r") as root:
            for j in range(0, num_steps):
                left_gripper = left_gripper_all[j]
                left_arm = left_arm_all[j]
                right_gripper = right_gripper_all[j]
                right_arm = right_arm_all[j]

                if j != num_steps - 1:
                    state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)
                    state = state.astype(np.float32)
                    qpos.append(state)

                    # Read only this frame from disk (avoids loading full episode images)
                    frame_bits = read_frame_images(root, cam_keys, j)
                    camera_high = cv2.imdecode(np.frombuffer(frame_bits["head_camera"], np.uint8), cv2.IMREAD_COLOR)
                    cam_high.append(cv2.resize(camera_high, (640, 480)))
                    camera_right_wrist = cv2.imdecode(np.frombuffer(frame_bits["right_camera"], np.uint8), cv2.IMREAD_COLOR)
                    cam_right_wrist.append(cv2.resize(camera_right_wrist, (640, 480)))
                    camera_left_wrist = cv2.imdecode(np.frombuffer(frame_bits["left_camera"], np.uint8), cv2.IMREAD_COLOR)
                    cam_left_wrist.append(cv2.resize(camera_left_wrist, (640, 480)))

                if j != 0:
                    actions.append(state)
                    left_arm_dim.append(left_arm.shape[0])
                    right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        # Explicit cleanup so memory is freed before next episode (avoids blow-up over 1000s of episodes)
        del qpos, actions, cam_high, cam_right_wrist, cam_left_wrist, left_arm_dim, right_arm_dim
        del left_gripper_all, left_arm_all, right_gripper_all, right_arm_all
        gc.collect()

        begin += 1

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    begin = 0
    begin = data_transform(
        os.path.join("../../data/", task_name, task_config, 'data'),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
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
