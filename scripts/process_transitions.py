import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from multiprocessing import Pool


def chunk_list(lst, n):
    # Calculate the size of each chunk
    chunk_size = len(lst) // n
    remainder = len(lst) % n

    # Create chunks
    chunks = [(i, lst[i * chunk_size:(i + 1) * chunk_size]) for i in range(n)]

    # Distribute the remainder elements evenly among the chunks
    for i in range(remainder):
        chunks[i][1].append(lst[n * chunk_size + i])

    return chunks


def load_pickle_with_progress(pickle_file, chunk_size=1024 * 1024 * 1024):  # Chunk size set to 1MB
    file_size = os.path.getsize(pickle_file)

    with open(pickle_file, 'rb') as file:
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
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


def process_episodes(episodes):
    idx, episodes = episodes
    use_symbolic_state = True
    for episode_i, episode in enumerate(tqdm(episodes, desc=f"#{idx}  Processing:")):
        obs_key = 'symbolic_state' if use_symbolic_state else 'observations'

        episode['actions'] = episode['actions'].astype(np.int8)
        episode[obs_key] = episode[obs_key].astype(np.int8)
        episode['rewards'] = episode['rewards'].astype(np.float32)

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
    return {'actions': actions,
            'observations': observations,
            'next_observations': next_observations,
            'rewards': rewards}


def save_transitions(dataset_dir):
    print('loading data')

    loaded_data = b''
    for chunk in load_pickle_with_progress(os.path.join(dataset_dir, 'trajectories.p'), chunk_size=1024 * 1024 * 1024):
        loaded_data += chunk
    episodes = pickle.loads(loaded_data)

    print('data loaded ')
    transition_data = {'actions': None,
                       'observations': None,
                       'next_observations': None,
                       'rewards': None}
    max_process = 8
    chunk_size = len(episodes) // max_process
    episode_chunks = [(e_i, episodes[i:i + chunk_size]) for e_i, i in
                      enumerate(range(0, len(episodes), chunk_size))]
    with Pool(max_process) as p:
        for x in p.map(process_episodes, episode_chunks):
            for k in x.keys():
                if transition_data[k] is None:
                    transition_data[k] = x[k]
                else:
                    transition_data[k] = np.concatenate((transition_data[k], x[k]))

    transition_data['actions'] = transition_data['actions'].astype(np.int8)
    transition_data['observations'] = transition_data['observations'].astype(np.int8)
    transition_data['next_observations'] = transition_data['next_observations'].astype(np.int8)
    transition_data['rewards'] = transition_data['rewards'].astype(np.float32)
    with open(os.path.join(dataset_dir, 'symbolic_state_transitions.p'), 'wb') as transitions_file:
        pickle.dump(transition_data, transitions_file)


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
