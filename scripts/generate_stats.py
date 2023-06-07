from sokoban_bazaar.dataset import get_dataset, pad_batch, _ENV_NAMES, _DATASET_NAMES
from markdownTable import markdownTable
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np

def generate_dataset_stats():
    df = pd.DataFrame(columns=['env_name', 'dataset_name', 'episodes'])
    for env_name in [ "Sokoban5x5-v0"]:
        for dataset_name in ["expert", "random", "expert-random"]:
            try:
                episode_dataset = get_dataset(env_name, dataset_name)
                episode_dataloader = DataLoader(
                    episode_dataset,
                    batch_size=256,
                    sampler=SequentialSampler(range(len(episode_dataset))),
                    num_workers=0,
                    collate_fn=pad_batch,
                )

                episode_score = {'max': -np.inf, 'min': np.inf, 'mean': 0}
                for batch in episode_dataloader:
                    episode_score['max'] = max(episode_score['max'], batch['returns_to_go'][:, 0, 0].max())
                    episode_score['min'] = min(episode_score['min'], batch['returns_to_go'][:, 0, 0].min())
                    episode_score['mean'] += batch['returns_to_go'][:, 0, 0].mean()
                episode_score['mean'] /= len(episode_dataloader)

                new_row = pd.Series({
                    'env_name': env_name,
                    'dataset_name': dataset_name,
                    'episodes': len(episode_dataset),
                    **{f'{k}-score': v for k, v in episode_score.items()}
                })
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            except:
                pass

    return df


if __name__ == '__main__':
    stats_df = generate_dataset_stats()
    print(stats_df)
