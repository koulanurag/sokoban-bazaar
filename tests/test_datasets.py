import pytest
import sokoban_bazaar as sb
from pytest_cases import parametrize, fixture_ref


@pytest.mark.parametrize('env_name,dataset_name',
                         [(_env_name, _dataset_name) for _dataset_name in
                          sb.dataset._DATASET_NAMES
                          for _env_name in sb.dataset._ENV_NAMES])
def test_dataset(env_name, dataset_name):
    from torch.utils.data import DataLoader

    episode_dataset = sb.dataset.get_dataset(env_name, dataset_name)
    episode_dataloader = DataLoader(
        episode_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=10,
        collate_fn=sb.dataset.pad_batch,
    )

    for batch in episode_dataloader:
        pass
