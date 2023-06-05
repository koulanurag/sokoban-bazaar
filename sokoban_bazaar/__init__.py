__ENV_NAMES = []
__DATASET_NAMES = []

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from sokoban_solver import PDDL, symbolic_state


def get_dataset(env_name, dataset_name):
    if env_name in __ENV_NAMES:
        raise ValueError()
    if dataset_name in __DATASET_NAMES:
        raise ValueError()

    _path = None
    episode_file_paths = os.listdir(_path)
    return EpisodeDataset(episode_file_paths)


def pad_batch(batch):
    # Sort the batch in descending order of sequence length
    batch = sorted(batch, key=lambda x: len(x["observations"]), reverse=True)

    # Pad the sequences to have the same length
    padded_batch = {
        k: torch.nn.utils.rnn.pad_sequence([_[k] for _ in batch], batch_first=True)
        for k in batch[0].keys()
    }
    return padded_batch


class EpisodeDataset(Dataset):
    """Episode dataset."""

    def __init__(self, dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path
        self.episode_files = os.listdir(self.dataset_dir_path)

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        episode = pickle.load(
            open(os.path.join(self.dataset_dir_path, self.episode_files[idx]), "rb")
        )

        observations = np.array(episode["observations"])
        actions = np.append(episode["actions"], episode["actions"][-1])
        rewards = np.append(episode["rewards"], 0)
        returns_to_go = np.append(episode["returns_to_go"], 0)
        timesteps = np.append(episode["timesteps"], episode["timesteps"][-1] + 1)

        # create tensors
        episode_info = {
            "observations": torch.FloatTensor(observations),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "returns_to_go": torch.FloatTensor(returns_to_go).unsqueeze(-1),
            "timesteps": torch.LongTensor(timesteps),
        }
        episode_info["attention_mask"] = torch.ones_like(episode_info["timesteps"])
        return episode_info


def generate_offline_dataset(
        env,
        domain_pddl_path,
        env_observation_mode,
        dataset_dir, max_episodes=1,
        episode_start_idx=0,
):
    total_step_count = 0
    done = True
    episode_count = 0

    # seed env
    env.action_space.seed(episode_start_idx)
    env.seed(episode_start_idx)

    with tqdm(total=max_episodes) as pbar:
        while episode_count < max_episodes:

            # ##########################################################################
            # Reset episode
            # ##########################################################################
            if done:
                episode_step_i = 0
                episode_info = defaultdict(lambda: [])
                env.reset()
                obs_img = env.render(mode=env_observation_mode)
                sym_state = symbolic_state(env.render(mode="tiny_rgb_array"))
                pddl = PDDL(
                    sym_state,
                    domain_pddl_path=domain_pddl_path,
                    problem_name=f"task-{random.randint(0, int(1e+10))}",
                    domain_name="sokoban",
                )
                plan = pddl.search_plan()

            # ##########################################################################
            # Step into environment
            # ##########################################################################
            action = plan.pop(0)  # pop action to be executed
            _, reward, done, info = env.step(action)
            next_obs_img = env.render(mode=env_observation_mode)
            total_step_count += 1
            episode_step_i += 1
            pbar.update(1)
            pbar.set_description(
                f"Episodes: {episode_count} " f"Steps: {total_step_count}"
            )

            # Convert image from (height, width, channel)
            # to (channel, height, width) for network compatibility.
            # Save rendered images as rendering allows different image modes
            episode_info["observations"].append(obs_img.transpose(2, 0, 1))
            episode_info["actions"].append(action)
            episode_info["rewards"].append(reward)
            episode_info["dones"].append(done)
            episode_info["timesteps"].append(episode_step_i)

            # ##########################################################################
            # Save episode
            # ##########################################################################
            if done:
                # store episode returns-to-go
                episode_info["returns_to_go"] = []
                return_to_go = 0
                for step_reward in episode_info["rewards"][::-1]:
                    return_to_go += step_reward
                    episode_info["returns_to_go"].append(return_to_go)
                episode_info["returns_to_go"] = episode_info["returns_to_go"][::-1]

                # store last observation
                episode_info["observations"].append(next_obs_img.transpose(2, 0, 1))

                pickle.dump(
                    dict(episode_info),
                    open(os.path.join(
                        dataset_dir,
                        f"episode_{episode_start_idx + episode_count}.p"),
                        "wb"
                    ),
                )
                episode_count += 1

            obs_img = next_obs_img
