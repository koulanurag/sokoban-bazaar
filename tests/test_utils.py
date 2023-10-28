import pytest
import gym
from sokoban_bazaar.utils import set_state, domain_pddl_path
from sokoban_bazaar.solver import PDDL, symbolic_state
import random
from tqdm import tqdm


@pytest.mark.parametrize(
    "env_name",
    [
        "gym_sokoban:Sokoban-small-v0",
        "gym_sokoban:Boxoban-Train-v0",
        # "gym_sokoban:Sokoban-small-v1",
        # "Sokoban5x5-v0",
        # "gym_sokoban:Sokoban-v2",
    ],
)
def test_set_state(env_name):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    step_env = gym.make(env_name)
    for episode_i in tqdm(range(50)):
        env.reset()
        eval_env = set_state(eval_env, env.render(mode="tiny_rgb_array"))

        # generate plan
        expert_plan = episode_i % 2 == 0
        if expert_plan:
            sym_state, _ = symbolic_state(env.render(mode="tiny_rgb_array"))
            pddl = PDDL(
                sym_state,
                domain_pddl_path=domain_pddl_path(env_name),
                problem_name=f"task-{random.randint(0, int(1e+10))}",
                domain_name="sokoban",
            )
            plan = pddl.search_plan()

        done = False
        while not done:
            if expert_plan:
                action = env.action_space.sample()
            else:
                action = plan.pop(0)

            obs, _, _, done = env.step(action)
            eval_obs, _, _, eval_done = eval_env.step(action)

            # try setting env for intermediate states
            set_state(step_env, env.render(mode="tiny_rgb_array"))
            step_env_obs = step_env.render(mode="rgb_array")

            assert done == eval_done
            assert (obs == eval_obs).all(), "observation don't match"
            assert (obs == step_env_obs).all(), "observation don't match"
