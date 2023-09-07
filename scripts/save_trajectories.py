import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

with open('file_list.txt', 'r') as files:
    processed_files = {k: None for k in files.read().split(", ")}

from multiprocessing import Pool


def process_files(file_paths):
    dataset_dir, id, file_paths = file_paths
    trajectories = []
    for file_path in tqdm(file_paths, desc=f"{id}"):
        file_path = os.path.join(dataset_dir, file_path)
        if 'episode' in file_path:
            try:
                with open(file_path, 'rb') as _file:
                    episode = pickle.load(_file)
                    trajectories.append({'observations': np.array(episode['observations']),
                                         'actions': np.array(episode['actions']).astype(np.int8),
                                         'rewards': np.array(episode['rewards']).astype(np.float32),
                                         'dones': np.array(episode['dones']),
                                         'timesteps': np.array(episode['timesteps']).astype(np.uint16),
                                         'symbolic_state': np.array(episode['symbolic_state']).astype(np.uint8),
                                         'returns_to_go': np.array(episode['returns_to_go']).astype(np.float32)})
            except:
                os.remove(file_path)
    return trajectories


def save_trajectories(dataset_dir):
    episode_files = os.listdir(dataset_dir)
    episode_files = episode_files[:len(episode_files) // 2]

    trajectories = []

    chunk_size = len(episode_files) // 5
    episode_file_chunks = [(dataset_dir, id, episode_files[i:i + chunk_size])
                           for id, i in enumerate(range(0, len(episode_files), chunk_size))]
    with Pool(5) as p:
        for x in p.map(process_files, episode_file_chunks):
            trajectories += x

    with open(os.path.join(dataset_dir, 'trajectories.p'), 'wb') as trajectories_file:
        pickle.dump(trajectories, trajectories_file)

    with open(os.path.join(dataset_dir, 'trajectories.p'), 'rb') as trajectories_file:
        pickle.load(trajectories_file)


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
    save_trajectories(dataset_dir)
