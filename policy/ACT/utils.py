import json
import numpy as np
import torch
import torch.distributed as dist
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import IPython

e = IPython.embed

LANG_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _build_lang_embeddings_cache(instructions_dir, episode_ids, cache_path):
    """Pre-compute sentence embeddings for all instructions and save to cache."""
    from sentence_transformers import SentenceTransformer

    print(f"Building language embedding cache for {len(episode_ids)} episodes...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_sentences = []
    episode_offsets = {}  # episode_id -> (start, count)
    for ep_id in sorted(episode_ids):
        json_path = os.path.join(instructions_dir, f"episode{ep_id}.json")
        if not os.path.isfile(json_path):
            episode_offsets[ep_id] = None
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        seen = data.get("seen", [])
        if len(seen) == 0:
            episode_offsets[ep_id] = None
            continue
        start = len(all_sentences)
        all_sentences.extend(seen)
        episode_offsets[ep_id] = (start, len(seen))

    if len(all_sentences) > 0:
        embeddings = model.encode(all_sentences, batch_size=256, show_progress_bar=True,
                                  convert_to_numpy=True)  # (N, 384)
        embeddings = torch.from_numpy(embeddings).float()
    else:
        embeddings = torch.zeros(0, LANG_DIM)

    # Split back into per-episode tensors
    cache = {}
    for ep_id, offset_info in episode_offsets.items():
        if offset_info is None:
            cache[ep_id] = torch.zeros(1, LANG_DIM)  # fallback zero embedding
        else:
            start, count = offset_info
            cache[ep_id] = embeddings[start:start + count]  # (count, 384)

    torch.save(cache, cache_path)
    print(f"Saved language embedding cache to {cache_path} ({len(cache)} episodes)")
    del model
    return cache


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len,
                 instructions_dir=None, lang_cond_type="none"):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.is_sim = None
        self.lang_cond_type = lang_cond_type
        self.lang_embeddings = None

        # Pre-compute language embeddings if needed
        if lang_cond_type != "none" and instructions_dir is not None:
            cache_path = os.path.join(dataset_dir, "lang_embeddings_cache.pt")
            if os.path.isfile(cache_path):
                print(f"Loading language embedding cache from {cache_path}")
                full_cache = torch.load(cache_path, map_location="cpu")
                # Only keep episodes relevant to this split
                self.lang_embeddings = {eid: full_cache[eid] for eid in episode_ids if eid in full_cache}
            else:
                # Need to build cache for ALL episodes (both train and val will share it)
                all_ep_ids = list(range(max(episode_ids) + 1))
                full_cache = _build_lang_embeddings_cache(instructions_dir, all_ep_ids, cache_path)
                self.lang_embeddings = {eid: full_cache[eid] for eid in episode_ids if eid in full_cache}

        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)  # 根据max_action_len初始化
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if self.lang_embeddings is not None and episode_id in self.lang_embeddings:
            embs = self.lang_embeddings[episode_id]  # (num_instructions, 384)
            idx = np.random.randint(len(embs))
            lang_embed = embs[idx]  # (384,)
            return image_data, qpos_data, action_data, is_pad, lang_embed

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats, max_action_len


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
              distributed=False, instructions_dir=None, lang_cond_type="none"):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # In distributed mode, ensure the lang embedding cache is built by rank 0
    # before any other rank tries to load it (avoids EOFError from partial writes).
    if distributed and lang_cond_type != "none" and instructions_dir is not None:
        cache_path = os.path.join(dataset_dir, "lang_embeddings_cache.pt")
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not os.path.isfile(cache_path):
            if rank == 0:
                all_ep_ids = list(range(num_episodes))
                _build_lang_embeddings_cache(instructions_dir, all_ep_ids, cache_path)
            # All ranks wait until rank 0 finishes writing the cache
            dist.barrier()

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len,
                                    instructions_dir=instructions_dir, lang_cond_type=lang_cond_type)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_action_len,
                                  instructions_dir=instructions_dir, lang_cond_type=lang_cond_type)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        )
    else:
        train_sampler = None
        val_sampler = None
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
        )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
