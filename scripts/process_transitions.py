import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Pool

from collections import defaultdict


def process_files(file_paths):
    dataset_dir, id, file_paths = file_paths
    transition_data = defaultdict(lambda: None)
    for file_path in tqdm(file_paths, desc=f"{id}"):
        file_path = os.path.join(dataset_dir, file_path)
        if 'episode' in file_path:
            with open(file_path, 'rb') as _file:
                episode = pickle.load(_file)

                episode = {'observations': np.array(episode['observations']),
                           'actions': np.array(episode['actions']).astype(np.int8),
                           'rewards': np.array(episode['rewards']).astype(np.float32),
                           'dones': np.array(episode['dones']),
                           'timesteps': np.array(episode['timesteps']).astype(np.uint16),
                           'symbolic_state': np.array(episode['symbolic_state']).astype(np.uint8),
                           'returns_to_go': np.array(episode['returns_to_go']).astype(np.float32)}

                for k in episode.keys():
                    if transition_data[k] is None:
                        if k in ['observations', 'symbolic_state']:
                            transition_data[k] = episode[k][:-1]
                            transition_data[f"next_{k}"] = episode[k][1:]
                        else:
                            transition_data[k] = episode[k]
                    else:
                        if k in ['observations', 'symbolic_state']:
                            transition_data[k] = np.concatenate((transition_data[k], episode[k][:-1]))
                            transition_data[f"next_{k}"] = np.concatenate((transition_data[f"next_{k}"], episode[k][1:]))
                        else:
                            transition_data[k] = np.concatenate((transition_data[k], episode[k]))

    return {k: v for k, v in transition_data.items()}


def save_transitions(dataset_dir):
    episode_files = os.listdir(dataset_dir)
    episode_files = episode_files[:len(episode_files) // 2]

    transition_data = defaultdict(lambda: None)

    max_processes = 10
    chunk_size = len(episode_files) // max_processes
    episode_file_chunks = [(dataset_dir, id, episode_files[i:i + chunk_size])
                           for id, i in enumerate(range(0, len(episode_files), chunk_size))]
    with Pool(max_processes) as p:
        for transition_data_chunk in p.map(process_files, episode_file_chunks):
            for k in transition_data_chunk.keys():
                if transition_data[k] is None:
                    transition_data[k] = transition_data_chunk[k]
                else:
                    transition_data[k] = np.concatenate((transition_data[k], transition_data_chunk[k]))

    transition_data = {k: v for k, v in transition_data.items()}
    with open(os.path.join(dataset_dir, 'transitions.p'), 'wb') as transitions_file:
        pickle.dump(transition_data, transitions_file)

    with open(os.path.join(dataset_dir, 'transitions.p'), 'rb') as transitions_file:
        pickle.load(transitions_file)



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
        default="expert",
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
