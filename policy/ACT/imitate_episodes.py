import os
import sys
import subprocess
import re
from datetime import timedelta

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pickle
import argparse
import wandb

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython

e = IPython.embed

WANDB_ENTITY = "far-wandb"
WANDB_PROJECT = "RoboTwin-conditioning"


def is_distributed():
    """Check if running in distributed mode (launched via torchrun)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank():
    if is_distributed():
        return int(os.environ["RANK"])
    return 0


def get_local_rank():
    if is_distributed():
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_world_size():
    if is_distributed():
        return int(os.environ["WORLD_SIZE"])
    return 1


def setup_ddp():
    """Initialize distributed process group."""
    if not is_distributed():
        return
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    if get_rank() == 0:
        print(f"DDP initialized: world_size={get_world_size()}")


def cleanup_ddp():
    """Destroy distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def main(args):
    set_seed(1)

    # Capture the training run start timestamp (used for wandb naming and eval video dirs)
    from datetime import datetime
    run_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize DDP if launched via torchrun
    distributed = is_distributed()
    if distributed:
        setup_ddp()
    rank = get_rank()
    local_rank = get_local_rank()

    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14  # yiheng
    lr_backbone = 1e-5
    backbone = "resnet18"
    lang_cond_type = args.get("lang_cond_type", "none") or "none"
    instructions_dir = args.get("instructions_dir")

    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "lang_cond_type": lang_cond_type,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq'],
        "distributed": distributed,
        "eval_task_name": args.get("eval_task_name"),
        "eval_task_config": args.get("eval_task_config", "demo_randomized"),
        "eval_episodes": args.get("eval_episodes", 10),
        "eval_step_lim": args.get("eval_step_lim"),
        "lang_cond_type": lang_cond_type,
        "instructions_dir": instructions_dir,
        "run_start_time": run_start_time,
    }

    if is_eval:
        ckpt_names = [f"policy_best.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    # Initialize wandb on rank 0
    if rank == 0:
        wandb_config = {
            "task_name": task_name,
            "policy_class": policy_class,
            "batch_size": batch_size_train,
            "num_epochs": num_epochs,
            "lr": args["lr"],
            "seed": args["seed"],
            "chunk_size": args.get("chunk_size"),
            "kl_weight": args.get("kl_weight"),
            "hidden_dim": args.get("hidden_dim"),
            "dim_feedforward": args.get("dim_feedforward"),
            "backbone": backbone,
            "state_dim": state_dim,
            "num_episodes": num_episodes,
            "camera_names": camera_names,
            "distributed": distributed,
            "world_size": get_world_size(),
            "effective_batch_size": batch_size_train * get_world_size(),
        }
        wandb_config["run_start_time"] = run_start_time
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"{task_name}_seed{args['seed']}_{run_start_time}",
            config=wandb_config,
            reinit=True,
        )

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train,
        batch_size_val, distributed=distributed,
        instructions_dir=instructions_dir, lang_cond_type=lang_cond_type,
    )

    # save dataset stats (rank 0 only)
    if rank == 0:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)

    # synchronize before training so all ranks see the saved stats
    if distributed:
        dist.barrier()

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint (rank 0 only)
    if rank == 0:
        ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")
        wandb.finish()

    if distributed:
        cleanup_ddp()


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1))
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )

        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
            )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def run_eval_subprocess(ckpt_dir, eval_task_name, eval_task_config, eval_episodes,
                        temporal_agg, eval_step_lim=None, seed=0, gpu_id=0,
                        run_start_time=None, lang_cond_type=None, lang_dim=None,
                        epoch=None):
    """Run evaluation as a subprocess using eval_policy.py and return success rate.

    This reuses the same eval infrastructure as eval.sh, spawning a separate
    process so that SAPIEN environment setup doesn't interfere with training.
    Output is streamed live so timestep progress and episode results are visible.
    """
    # Determine repo root (this file lives at policy/ACT/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../.."))

    # ckpt_dir must be absolute so the subprocess can find it
    abs_ckpt_dir = os.path.abspath(ckpt_dir)

    cmd = [
        sys.executable, "-u", "script/eval_policy.py",
        "--config", "policy/ACT/deploy_policy.yml",
        "--overrides",
        "--task_name", eval_task_name,
        "--task_config", eval_task_config,
        "--ckpt_setting", eval_task_config,
        "--ckpt_dir", abs_ckpt_dir,
        "--seed", str(seed),
        "--test_num", str(eval_episodes),
        "--eval_video_log", "True",
    ]
    if temporal_agg:
        cmd.extend(["--temporal_agg", "true"])
    if eval_step_lim is not None:
        cmd.extend(["--eval_step_lim", str(eval_step_lim)])
    if run_start_time is not None:
        cmd.extend(["--run_start_time", run_start_time])
    if lang_cond_type and lang_cond_type != "none":
        cmd.extend(["--lang_cond_type", lang_cond_type])
    if lang_dim is not None:
        cmd.extend(["--lang_dim", str(lang_dim)])
    if epoch is not None:
        cmd.extend(["--eval_epoch", str(epoch)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'='*60}")
    print(f"Running evaluation: {eval_episodes} episodes on {eval_task_name}")
    print(f"Checkpoint dir: {abs_ckpt_dir}")
    if eval_step_lim is not None:
        print(f"Step limit per episode: {eval_step_lim}")
    print(f"{'='*60}\n", flush=True)

    collected_output = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=repo_root, env=env,
        )

        # Stream output line-by-line so the user sees progress in real time
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            collected_output.append(line)

        proc.wait(timeout=7200)

        full_output = "\n".join(collected_output)
        # Strip ANSI colour codes before parsing
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_output = ansi_escape.sub('', full_output)

        # Parse the last "Success rate: X/Y" line (printed after every episode)
        matches = re.findall(r'Success rate: (\d+)/(\d+)', clean_output)
        if matches:
            successes, total = int(matches[-1][0]), int(matches[-1][1])
            success_rate = successes / total if total > 0 else 0.0
            print(f"\nEvaluation complete: {successes}/{total} = {success_rate*100:.1f}%")
            return success_rate
        else:
            print("Warning: Could not parse eval results from output")
            if proc.returncode != 0:
                print(f"Eval process exited with code {proc.returncode}")
            return None
    except subprocess.TimeoutExpired:
        print("Warning: Evaluation timed out")
        proc.kill()
        return None
    except Exception as exc:
        print(f"Warning: Evaluation failed with error: {exc}")
        return None


def forward_pass(data, policy, device=None):
    if len(data) == 5:
        image_data, qpos_data, action_data, is_pad, lang_embed = data
    else:
        image_data, qpos_data, action_data, is_pad = data
        lang_embed = None
    if device is None:
        device = torch.device("cuda")
    image_data, qpos_data, action_data, is_pad = (
        image_data.to(device),
        qpos_data.to(device),
        action_data.to(device),
        is_pad.to(device),
    )
    if lang_embed is not None:
        lang_embed = lang_embed.to(device)
    return policy(qpos_data, image_data, action_data, is_pad, lang_embed=lang_embed)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    distributed = config.get("distributed", False)

    rank = get_rank()
    local_rank = get_local_rank()

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    # Wrap with DDP for multi-GPU training
    if distributed:
        ddp_policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_policy = policy

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    epoch_iter = tqdm(range(num_epochs)) if rank == 0 else range(num_epochs)
    for epoch in epoch_iter:
        if rank == 0:
            print(f"\nEpoch {epoch}")

        # Set epoch for distributed sampler (ensures proper shuffling each epoch)
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)

        # validation
        with torch.inference_mode():
            ddp_policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, ddp_policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                # Save state_dict from the unwrapped policy (without DDP "module." prefix)
                unwrapped = ddp_policy.module if distributed else ddp_policy
                best_ckpt_info = (epoch, min_val_loss, deepcopy(unwrapped.state_dict()))
        if rank == 0:
            print(f"Val loss:   {epoch_val_loss:.5f}")
            summary_string = ""
            for k, v in epoch_summary.items():
                summary_string += f"{k}: {v.item():.3f} "

        # training
        ddp_policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, ddp_policy, device)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary["loss"]
        if rank == 0:
            print(f"Train loss: {epoch_train_loss:.5f}")
            summary_string = ""
            for k, v in epoch_summary.items():
                summary_string += f"{k}: {v.item():.3f} "

            # Log to wandb
            wandb_log = {"epoch": epoch}
            for k, v in epoch_summary.items():
                wandb_log[f"train/{k}"] = v.item()
            val_summary = validation_history[-1]
            for k, v in val_summary.items():
                wandb_log[f"val/{k}"] = v.item()
            wandb_log["val/best_loss"] = min_val_loss.item() if hasattr(min_val_loss, 'item') else float(min_val_loss)
            wandb.log(wandb_log, step=epoch)

        if (epoch + 1) % config['save_freq'] == 0 and rank == 0:
            unwrapped = ddp_policy.module if distributed else ddp_policy
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            torch.save(unwrapped.state_dict(), ckpt_path)

            # Also save as policy_last.ckpt (used by the eval/deploy pipeline)
            last_ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            torch.save(unwrapped.state_dict(), last_ckpt_path)

            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

            # Run evaluation if configured
            eval_task_name = config.get('eval_task_name')
            if eval_task_name:
                success_rate = run_eval_subprocess(
                    ckpt_dir=ckpt_dir,
                    eval_task_name=eval_task_name,
                    eval_task_config=config.get('eval_task_config', 'demo_randomized'),
                    eval_episodes=config.get('eval_episodes', 10),
                    temporal_agg=config.get('temporal_agg', False),
                    eval_step_lim=config.get('eval_step_lim'),
                    seed=0,
                    gpu_id=local_rank,
                    run_start_time=config.get('run_start_time'),
                    lang_cond_type=config.get('lang_cond_type'),
                    lang_dim=config.get('policy_config', {}).get('lang_dim'),
                    epoch=epoch + 1,
                )
                if success_rate is not None:
                    wandb.log({"eval/success_rate": success_rate}, step=epoch)

        # Synchronize all ranks after checkpoint saving / eval so that non-zero
        # ranks don't race ahead into the next epoch's DDP collectives while
        # rank 0 is still busy (which would cause an NCCL timeout).
        if distributed:
            dist.barrier()

    if rank == 0:
        unwrapped = ddp_policy.module if distributed else ddp_policy
        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        torch.save(unwrapped.state_dict(), ckpt_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

        # save training curves
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--state_dim", action="store", type=int, help="state dim", required=True)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # Language conditioning
    parser.add_argument("--lang_cond_type", type=str, default="none",
                        choices=["none", "film", "token"],
                        help="Language conditioning variant (default: none)")
    parser.add_argument("--instructions_dir", type=str, default=None,
                        help="Path to directory with per-episode instruction JSON files")

    # Evaluation during training (optional)
    parser.add_argument("--eval_task_name", type=str, default=None,
                        help="Task name for eval (e.g. place_object_stand). If set, eval runs every save_freq epochs.")
    parser.add_argument("--eval_task_config", type=str, default="demo_randomized",
                        help="Task config yml name for eval (default: demo_randomized)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of eval episodes to run (default: 10)")
    parser.add_argument("--eval_step_lim", type=int, default=None,
                        help="Max timesteps per eval episode (default: use task-specific limit)")

    main(vars(parser.parse_args()))
