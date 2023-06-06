from sokoban_bazaar.dataset import get_dataset, pad_batch, _ENV_NAMES, _DATASET_NAMES
from markdownTable import markdownTable
import pandas as pd


def generate_dataset_stats(format="markdown"):
    df = pd.DataFrame(columns=['env_name', 'dataset_name', 'episodes'])
    for env_name in _ENV_NAMES:
        for dataset_name in ["expert"]:
            try:
                new_row = pd.Series({
                    'env_name': env_name,
                    'dataset_name': dataset_name,
                    'episodes': len(get_dataset(env_name, dataset_name))
                })
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            except:
                pass

    return df


if __name__ == '__main__':
    mark_down = generate_dataset_stats()
    print(mark_down)
