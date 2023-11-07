from pathlib import Path
import gym
from ..utils import set_state
from torch.utils.data import DataLoader, WeightedRandomSampler
from gym_sokoban.envs import SokobanEnv
from tqdm import tqdm
from collections import defaultdict

_ENV_NAMES = [
    # "gym_sokoban:Sokoban-small-v0",
    # "gym_sokoban:Sokoban-small-v1",
    # "gym_sokoban:Sokoban-v2",
    # "Sokoban5x5-v0",
    # "gym_sokoban:Sokoban-large-v0",
    # "gym_sokoban:Sokoban-large-v1",
    "gym_sokoban:Boxoban-Train-v0",
]
_DATASET_NAMES = [
    #    "random",
    "expert",
    # 'medium',
    # 'medium-expert',
    #    "expert-random",
]

import os
import pickle

import numpy as np
import torch


def __root_dir():
    return os.environ.get(
        "SOKOBAN_DATASET_ROOT_PATH",
        default=os.path.join(Path.home(), ".sokoban-datasets"),
    )


def __load_pickle_with_progress(
    pickle_file, chunk_size=1024 * 1024 * 1024, desc=None
):  # Chunk size set to 1MB
    file_size = os.path.getsize(pickle_file)

    with open(pickle_file, "rb") as file:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc
        ) as pbar:
            try:
                while True:
                    data = file.read(chunk_size)
                    if not data:
                        break
                    pbar.update(len(data))
                    yield data
            except KeyboardInterrupt:
                pbar.close()
                print("Loading interrupted by user.")
                exit(1)
            except Exception as e:
                pbar.close()
                print(f"An error occurred: {str(e)}")
                exit(1)


def _load_pickle_with_progress(pickle_file, chunk_size=1024 * 1024 * 1024, desc=None):
    loaded_data = b""
    for chunk in __load_pickle_with_progress(pickle_file, chunk_size, desc):
        loaded_data += chunk
    return pickle.loads(loaded_data)


def get_trajectories(env_name, dataset_name, dataset_size=None, future_context=False, chunk_size=1024**3, with_embedding=False):
    if env_name not in _ENV_NAMES:
        raise ValueError()
    if dataset_name not in ["expert", "random", "expert-random"]:
        raise ValueError()

    if dataset_name in ["expert", "random", "expert-random"]:
        trajectories = []
        sub_dataset_names = dataset_name.split("-")
        weights = []
        for sub_dataset_name in sub_dataset_names:

            if future_context and not with_embedding:
                _trajectories_path = os.path.join(
                    __root_dir(),
                    env_name,
                    "tiny_rgb_array",
                    sub_dataset_name,
                    "branched-trajectories",
                    "trajectories.p",
                )
            elif with_embedding:
                _trajectories_path = os.path.join(
                    __root_dir(),
                    env_name,
                    "tiny_rgb_array",
                    sub_dataset_name,
                    "future-embedding",
                    "trajectories.p",
                )
            else:
                _trajectories_path = os.path.join(
                    __root_dir(),
                    env_name,
                    "tiny_rgb_array",
                    sub_dataset_name,
                    "trajectories.p",
                )

            sub_dataset = _load_pickle_with_progress(
                _trajectories_path,
                chunk_size=chunk_size,
                desc=f"Loading Dataset {sub_dataset_name}",
            )
            sub_dataset_size = len(sub_dataset)
            trajectories += sub_dataset[:sub_dataset_size]
            weights += (
                np.ones(len(sub_dataset[:sub_dataset_size]))
                * 1
                / len(sub_dataset[:sub_dataset_size])
            ).tolist()
    else:
        raise ValueError()

    return trajectories, WeightedRandomSampler(weights, len(weights), replacement=True)


def get_transitions(env_name, dataset_name, dataset_size=None, chunk_size=1024**3):
    if env_name not in _ENV_NAMES:
        raise ValueError()
    if dataset_name not in ["expert", "random", "expert-random"]:
        raise ValueError()

    if dataset_name in ["expert", "random", "expert-random"]:
        transitions = defaultdict(lambda: None)
        sub_dataset_names = dataset_name.split("-")
        weights = []
        for sub_dataset_name in sub_dataset_names:
            _trajectories_path = os.path.join(
                root_dir(),
                env_name,
                "tiny_rgb_array",
                sub_dataset_name,
                "transitions.p",
            )
            sub_dataset = load_pickle_with_progress(
                _trajectories_path,
                chunk_size=chunk_size,
                desc=f"Loading Dataset {sub_dataset_name}",
            )
            sub_dataset_size = len(sub_dataset["actions"])

            for k, v in sub_dataset.items():
                if transitions[k] is None:
                    transitions[k] = v[:sub_dataset_size]
                else:
                    transitions[k] = np.concatenate(
                        (transitions[k], v[:sub_dataset_size])
                    )
            weights += (
                np.ones(len(sub_dataset["actions"][:sub_dataset_size]))
                * 1
                / len(sub_dataset["actions"][:sub_dataset_size])
            ).tolist()
    else:
        raise ValueError()

    return transitions, WeightedRandomSampler(weights, len(weights), replacement=True)


def get_test_envs(env_name):
    if env_name not in _ENV_NAMES:
        raise ValueError()

    if "Boxoban" in env_name:
        return get_boxban_test_envs(env_name)
    else:
        with open(
            os.path.join(__root_dir(), env_name, "test_states.p"), "rb"
        ) as test_files:
            test_states = pickle.load(test_files)

        test_envs = []
        for _, test_state_info in test_states.items():
            env = SokobanCustomResetEnv(reset=False)
            env.reset(tiny_rgb_state=test_state_info["tiny_rgb_array"])
            test_state_info["hardness-level"] = 0
            test_envs.append((env, test_state_info))

    return test_envs


def get_boxban_test_envs(env_name):
    assert env_name in ["gym_sokoban:Boxoban-Train-v0"]

    with open(os.path.join(__root_dir(), env_name, "test_states.p"), "rb") as test_files:
        test_states = pickle.load(test_files)

    test_envs = []
    for start_idx in range(0, 1000, 100):
        for test_state_info in test_states[start_idx + 70 : start_idx + 100]:
            env = SokobanCustomResetEnv(reset=False)
            env.reset(tiny_rgb_state=test_state_info["tiny_rgb_array"])
            # set_state(env, test_state_info['tiny_rgb_array'])
            test_state_info["hardness-level"] = start_idx
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

    def reset(self, second_player=False, render_mode="rgb_array", tiny_rgb_state=None):
        if tiny_rgb_state is None:
            return super().reset()

        else:
            return self.reset_to_state(
                tiny_rgb_state=tiny_rgb_state,
                second_player=second_player,
                render_mode=render_mode,
            )

    def reset_to_state(
        self, tiny_rgb_state, second_player=False, render_mode="rgb_array"
    ):
        from ..solver import symbolic_state

        sym_state, info = symbolic_state(tiny_rgb_state)
        self.room_fixed = info["room_fixed"]
        self.room_state = info["room_state"]
        self.box_mapping = info["box_mapping"]

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self._has_reset = True

        return self.render(render_mode)
