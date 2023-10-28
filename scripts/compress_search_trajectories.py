import pickle
from pathlib import Path
import os

root_path = os.environ.get(
    "SOKOBAN_DATASET_ROOT_PATH", default=os.path.join(Path.home(), ".sokoban-datasets")
)


def compress_search_trajectories():
    env_name = "gym_sokoban:Boxoban-Train-v0"
    dataset_name = "expert"
    save_dir = os.path.join(
        root_path, env_name, "tiny_rgb_array", dataset_name, "branched-trajectories"
    )
    os.makedirs(save_dir, exist_ok=True)

    episodes = []
    for file_name in os.listdir(save_dir):
        episode = pickle.load(open(os.path.join(save_dir, file_name), 'rb'))
        episodes.append(episode)

    pickle.dump(episodes, open(os.path.join(save_dir, f"trajectories.p"), "wb"))


if __name__ == "__main__":
    compress_search_trajectories()
