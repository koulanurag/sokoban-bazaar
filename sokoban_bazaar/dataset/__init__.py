from pathlib import Path
import gym
from ..utils import set_state
from torch.utils.data import DataLoader, WeightedRandomSampler
from gym_sokoban.envs import SokobanEnv
from tqdm import tqdm

_ENV_NAMES = [
    # "gym_sokoban:Sokoban-small-v0",
    "gym_sokoban:Sokoban-small-v1",
    "gym_sokoban:Sokoban-v2",
    "Sokoban5x5-v0",
    "gym_sokoban:Sokoban-large-v0",
    "gym_sokoban:Sokoban-large-v1",
    "gym_sokoban:Boxoban-Train-v0"]
_DATASET_NAMES = ['random',
                  'expert',
                  # 'medium',
                  # 'medium-expert',
                  'expert-random']

import os
import pickle

import numpy as np
import torch


def root_dir():
    return os.environ.get('SOKOBAN_DATASET_ROOT_PATH',
                          default=os.path.join(Path.home(), ".sokoban-datasets"))


def get_trajectories(env_name, dataset_name, dataset_size=None):
    if env_name not in _ENV_NAMES:
        raise ValueError()
    if dataset_name not in ['expert', 'random', 'expert-random']:
        raise ValueError()

    if dataset_name in ['expert', 'random', 'expert-random']:
        trajectories = []
        sub_dataset_names = dataset_name.split("-")
        weights = []
        for sub_dataset_name in sub_dataset_names:
            with open(os.path.join(root_dir(), env_name,
                                   'tiny_rgb_array', sub_dataset_name, 'trajectories.p'),
                      'rb') as trajectories_file:

                sub_dataset = pickle.load(trajectories_file)

                # if dataset_size is None:
                #     sub_dataset_size = len(sub_dataset) // len(sub_dataset_names)
                # else:
                #     sub_dataset_size = dataset_size // len(sub_dataset_names)
                #
                sub_dataset_size = len(sub_dataset)
                trajectories += sub_dataset[:sub_dataset_size]
                weights += (np.ones(len(sub_dataset[:sub_dataset_size]))
                            * 1 / len(sub_dataset[:sub_dataset_size])).tolist()
    else:
        raise ValueError()

    return trajectories, \
        WeightedRandomSampler(weights, len(weights), replacement=True)

def get_transitions(env_name, dataset_name, dataset_size=None):
    if env_name not in _ENV_NAMES:
        raise ValueError()
    if dataset_name not in ['expert', 'random', 'expert-random']:
        raise ValueError()

    if dataset_name in ['expert', 'random', 'expert-random']:
        transitions = []
        sub_dataset_names = dataset_name.split("-")
        weights = []
        for sub_dataset_name in sub_dataset_names:
            with open(os.path.join(root_dir(), env_name, 'tiny_rgb_array', sub_dataset_name, 'transitions.p'),
                      'rb') as trajectories_file:

                sub_dataset = pickle.load(trajectories_file)

                # if dataset_size is None:
                #     sub_dataset_size = len(sub_dataset) // len(sub_dataset_names)
                # else:
                #     sub_dataset_size = dataset_size // len(sub_dataset_names)
                #
                sub_dataset_size = len(sub_dataset)
                transitions += sub_dataset[:sub_dataset_size]
                weights += (np.ones(len(sub_dataset[:sub_dataset_size]))
                            * 1 / len(sub_dataset[:sub_dataset_size])).tolist()
    else:
        raise ValueError()

    return transitions, \
        WeightedRandomSampler(weights, len(weights), replacement=True)



def get_test_envs(env_name):
    if env_name not in _ENV_NAMES:
        raise ValueError()

    if 'Boxoban' in env_name:
        return get_boxban_test_envs(env_name)
    else:
        with open(os.path.join(root_dir(), env_name, 'test_states.p'), 'rb') as test_files:
            test_states = pickle.load(test_files)

        test_envs = []
        for _, test_state_info in test_states.items():
            env = SokobanCustomResetEnv(reset=False)
            env.reset(tiny_rgb_state=test_state_info['tiny_rgb_array'])
            test_state_info['hardness-level'] = 0
            test_envs.append((env, test_state_info))

    return test_envs


def get_boxban_test_envs(env_name):
    assert env_name in ['gym_sokoban:Boxoban-Train-v0']

    with open(os.path.join(root_dir(), env_name, 'test_states.p'), 'rb') as test_files:
        test_states = pickle.load(test_files)

    test_envs = []
    for start_idx in range(0, 1000, 100):
        for test_state_info in test_states[start_idx + 70: start_idx + 100]:
            env = SokobanCustomResetEnv(reset=False)
            env.reset(tiny_rgb_state=test_state_info['tiny_rgb_array'])
            # set_state(env, test_state_info['tiny_rgb_array'])
            test_state_info['hardness-level'] = start_idx
            test_envs.append((env, test_state_info))

    return test_envs


class Trajectories:
    """Episode dataset."""

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.index = 0

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(os.path.join(self.episode_files[idx]), "rb") as data_file:
            episode = pickle.load(data_file)
        observations = np.array(episode["observations"])
        actions = np.array(episode["actions"])
        rewards = np.array(episode["rewards"])
        returns_to_go = np.array(episode["returns_to_go"])
        timesteps = np.array(episode["timesteps"])

        # create tensors
        episode_info = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
        }
        return episode_info

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        value = self[self.index]
        self.index += 1
        return value


class SokobanCustomResetEnv(SokobanEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, second_player=False, render_mode='rgb_array', tiny_rgb_state=None):
        if tiny_rgb_state is None:
            return super().reset()

        else:
            return self.reset_to_state(tiny_rgb_state=tiny_rgb_state,
                                       second_player=second_player,
                                       render_mode=render_mode)

    def reset_to_state(self, tiny_rgb_state, second_player=False, render_mode='rgb_array'):
        from ..solver import symbolic_state

        sym_state, info = symbolic_state(tiny_rgb_state)
        self.room_fixed = info['room_fixed']
        self.room_state = info['room_state']
        self.box_mapping = info['box_mapping']

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self._has_reset = True

        return self.render(render_mode)
