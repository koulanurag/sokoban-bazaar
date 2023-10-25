import pytest
from sokoban_bazaar.solver import PDDL, symbolic_state
from sokoban_bazaar.utils import domain_pddl_path
import gym
import random


@pytest.mark.parametrize("env_name", ["gym_sokoban:Sokoban-small-v0"])
def test_solver(env_name):
    env = gym.make(env_name)
    env.reset()
    sym_state, _ = symbolic_state(env.render(mode="tiny_rgb_array"))

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
        _, _, _, info = env.step(action)

    # verify if solved
    assert info["all_boxes_on_target"]
