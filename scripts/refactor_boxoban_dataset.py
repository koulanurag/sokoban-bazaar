import pickle

from sokoban_bazaar.solver import symbolic_state
from tqdm import tqdm
import os
import argparse


def process_episode_files(episode_start_idx, max_episodes):
    root_dir = '/mnt/dt_sokoban/datasets/gym_sokoban:Boxoban-Train-v0/tiny_rgb_array/expert'
    # root_dir = '/Users/anuragkoul/.sokoban-datasets/gym_sokoban:Boxoban-Train-v0/tiny_rgb_array/expert'

    episode_files = os.listdir(root_dir)[episode_start_idx:episode_start_idx + max_episodes]
    for episode_file_path in tqdm(episode_files):
        episode_file_path = os.path.join(root_dir, episode_file_path)
        try:
            with open(episode_file_path, 'rb') as episode_file:
                episode_info = pickle.load(episode_file)
        except Exception as e:
            print(f'error loading file {episode_file_path}')
        finally:
            episode_info['symbolic_state'] = []
            for step_i in range(len(episode_info['observations'])):
                _, obs_info = symbolic_state(episode_info['observations'][step_i].swapaxes(0, 1).swapaxes(1, 2))
                episode_info['symbolic_state'].append(obs_info['true_state'].flatten())
            pickle.dump(episode_info, open(episode_file_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser("Combinatorial Tasks with Decision Transformers ")
    parser.add_argument(
        "--episode-start-idx",
        default=0,
        type=int,
        help="max number steps for data collection",
    )
    parser.add_argument(
        "--max-episodes",
        default=1000,
        type=int,
        help="max number steps for data collection",
    )

    # process arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    process_episode_files(args.episode_start_idx, args.max_episodes)
