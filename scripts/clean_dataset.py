import pickle

from sokoban_bazaar.dataset import get_dataset
from tqdm import tqdm
import os
import argparse


def generate_dataset_stats(episode_start_idx, max_episodes):
    removed_files = []
    missing_files = []
    for env_name in ['gym_sokoban:Boxoban-Train-v0']:
        for dataset_name in ['expert']:
            print(env_name, dataset_name)

            try:
                episode_dataset, _ = get_dataset(env_name, dataset_name)
                pbar = tqdm(range(episode_start_idx, episode_start_idx + max_episodes))
                for file_i in pbar:
                    try:
                        file = episode_dataset.episode_files[file_i]
                        try:
                            with open(file, 'rb') as _file:
                                _episode_info = pickle.load(_file)
                        except:
                            os.remove(file)
                            removed_files.append(file)
                    except:
                        missing_files.append(file_i)
                    pbar.set_description(f"Removed Files: {len(removed_files)} | Missing Files: {len(missing_files)}")

            except Exception as e:
                print(e)

    return removed_files, missing_files


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
    removed_files, missing_files = generate_dataset_stats(args.episode_start_idx, args.max_episodes)
    for x in removed_files:
        print(x)
    print('----')
    for x in missing_files:
        print(x)
