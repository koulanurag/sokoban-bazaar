from pathlib import Path
import gym
from ..utils import set_state

_ENV_NAMES = ["gym_sokoban:Sokoban-small-v0",
              "gym_sokoban:Sokoban-small-v1",
              "gym_sokoban:Sokoban-v2",
              "Sokoban5x5-v0",
              "gym_sokoban:Sokoban-large-v0",
              "gym_sokoban:Sokoban-large-v1"]
_DATASET_NAMES = ['random',
                  'expert',
                  'medium',
                  'medium-expert',
                  'expert-random']

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def root_dir():
    return os.environ.get('SOKOBAN_DATASET_ROOT_PATH',
                          default=os.path.join(Path.home(), ".sokoban-datasets"))


def get_dataset(env_name, dataset_name, dataset_size=None):
    if env_name not in _ENV_NAMES:
        raise ValueError()
    if dataset_name not in _DATASET_NAMES:
        raise ValueError()

    _root_dir = root_dir()
    if dataset_name in ['expert', 'random', 'expert-random']:
        episode_file_paths = []
        sub_dataset_names = dataset_name.split("-")
        for sub_dataset_name in sub_dataset_names:
            dataset_dir = os.path.join(_root_dir, env_name, 'tiny_rgb_array',
                                       sub_dataset_name)
            dataset_files = os.listdir(dataset_dir)

            if dataset_size is None:
                sub_dataset_size = len(dataset_files) // len(sub_dataset_names)
            else:
                sub_dataset_size = dataset_size // len(sub_dataset_names)

            episode_file_paths += [os.path.join(dataset_dir, file)
                                   for file in dataset_files[:sub_dataset_size]]
    else:
        raise ValueError()

    return EpisodeDataset(episode_file_paths)


def get_test_envs(env_name):
    if env_name not in _ENV_NAMES:
        raise ValueError()

    with open(os.path.join(root_dir(), env_name, 'test_states.p'), 'rb') as test_files:
        test_states = pickle.load(test_files)

    test_envs = []
    for _, test_state_info in test_states.items():
        env = gym.make(env_name)
        set_state(env, test_state_info['tiny_rgb_array'])
        test_envs.append(env)

    return test_envs


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

    def __init__(self, episode_file_paths):
        self.episode_files = episode_file_paths

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        episode = pickle.load(open(os.path.join(self.episode_files[idx]), "rb"))

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
