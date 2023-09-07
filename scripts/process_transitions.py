import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


def load_pickle_with_progress(pickle_file):
    file_size = os.path.getsize(pickle_file)

    with open(pickle_file, 'rb') as file:
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                data = file.read(1024)
                if not data:
                    break
                pbar.update(len(data))
                yield data


def save_transitions(dataset_dir):
    use_symbolic_state = True
    print('loading data')

    loaded_data = b''
    for chunk in load_pickle_with_progress(os.path.join(dataset_dir, 'trajectories.p')):
        loaded_data += chunk
    episodes = pickle.loads(loaded_data)

    # with open(os.path.join(dataset_dir, 'trajectories.p'), 'rb') as trajectories_file:
    #     episodes = pickle.load(trajectories_file)

    print('data loaded ')
    for episode_i, episode in enumerate(tqdm(episodes, desc="Transition Dataset Processing:")):
        obs_key = 'symbolic_state' if use_symbolic_state else 'observations'

        if episode_i > 0:
            actions = np.concatenate((actions, episode['actions']))
            observations = np.concatenate((observations, episode[obs_key][:-1]))
            next_observations = np.concatenate((next_observations, episode[obs_key][1:]))
            rewards = np.concatenate((rewards, episode['rewards']))
        else:
            actions = episode['actions']
            observations = episode[obs_key][:-1]
            next_observations = episode[obs_key][1:]
            rewards = episode['rewards']

    with open(os.path.join(dataset_dir, 'symbolic_state_transitions.p'), 'wb') as transitions_file:
        pickle.dump({'actions': actions,
                     'observations': observations,
                     'next_observations': next_observations,
                     'rewards': rewards}, transitions_file)


def get_args():
    parser = argparse.ArgumentParser("Combinatorial Tasks with Decision Transformers ")
    parser.add_argument(
        "--job",
        default="train",
        type=str,
        choices=["train", "eval", "generate-offline-data"],
        help="job to be performed",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(Path.home(), ".sokoban-datasets")
    )

    # env-args
    env_args = parser.add_argument_group("data generation args")
    env_args.add_argument(
        "--env-name",
        default="gym_sokoban:Boxoban-Train-v0",
        help="name of the environment",
    )
    env_args.add_argument(
        "--env-observation-mode",
        default="tiny_rgb_array",
        choices=["rgb_array", "tiny_rgb_array"],
        help="render( image mode) for the environment",
    )

    # data-generation args
    data_generation_args = parser.add_argument_group("data generation args")
    data_generation_args.add_argument(
        "--max-episodes",
        default=1000,
        type=int,
        help="max number steps for data collection",
    )
    data_generation_args.add_argument(
        "--dataset-quality",
        default="random",
        type=str,
        help="max number steps for data collection",
        choices=[
            "expert",
            "random"
        ],
    )
    data_generation_args.add_argument(
        "--episode-start-idx",
        default=0,
        type=int,
        help="max number steps for data collection",
    )
    data_generation_args.add_argument(
        "--source-file-idx",
        default=0,
        type=int,
        help="max number steps for data collection",
    )

    # process arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    dataset_dir = os.path.join(args.dataset_dir, args.env_name,
                               args.env_observation_mode, args.dataset_quality)

    os.makedirs(dataset_dir, exist_ok=True)
    save_transitions(dataset_dir)
