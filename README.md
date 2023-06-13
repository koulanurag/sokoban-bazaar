# sokoban-bazaar

A bazaar of sokokan datasets and solver

# Installation

   ```bash
   git clone https://github.com/koulanurag/sokoban-bazaar.git
   cd sokoban-bazaar
   pip install -e .
   ```

# Usage

1. Train Datasets

```python
from torch.utils.data import DataLoader
from sokoban_bazaar.dataset import get_dataset, pad_episodic_batch

episode_dataset, weighted_sampler = get_dataset(env_name="gym_sokoban:Sokoban-v2", dataset_name="expert")
episode_dataloader = DataLoader(
    episode_dataset,
    batch_size=256,
    num_workers=0,
    sampler=weighted_sampler,
    collate_fn=pad_episodic_batch
)

for batch in episode_dataloader:
    pass

```

2. Test Environments (in-development)

```python
from sokoban_bazaar.dataset import get_test_envs

test_envs = get_test_envs(env_name="gym_sokoban:Sokoban-v2", dataset_name="expert")

for env in test_envs:
    done = False
    episode_score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_score += reward

    print(f"Episode Score: {episode_score}| {'Success' if info['all_boxes_on_target'] else 'Failure'}")
```

3. Solver

```python
import gym
import random
from sokoban_bazaar.solver import PDDL, symbolic_state
from sokoban_bazaar.utils import domain_pddl_path

# create env
env_name = "gym_sokoban:Sokoban-v2"
env = gym.make(env_name)
env.reset()
sym_state = symbolic_state(env.render(mode="tiny_rgb_array"))

# generate plan
pddl = PDDL(
    sym_state,
    domain_pddl_path=domain_pddl_path(env_name),
    problem_name=f"task-{random.randint(0, int(1e+10))}",
    domain_name="sokoban",
)
plan = pddl.search_plan()

# execute plan
for action in plan:
    _, _, done, info = env.step(action)
    if done:
        break

print("Success" if info["all_boxes_on_target"] else "Failure")
```

## Testing:

- Install: ```pip install -e ".[test]" ```
- Run: ```pytest```

## Contact

If you have any questions or suggestions , you can contact me at koulanurag@gmail.com or open an issue on this GitHub repository.
