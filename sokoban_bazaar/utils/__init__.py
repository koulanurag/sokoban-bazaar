import os
from pathlib import Path
import numpy as np
from ..solver import symbolic_state


def domain_pddl_path(env_name):
    assets_dir = os.path.join(Path(os.path.dirname(__file__)).parent, 'assets')
    return os.path.join(assets_dir, 'sokoban_domain.pddl')


def set_state(env, tiny_rgb_observation):
    sym_state, info = symbolic_state(tiny_rgb_observation)
    env.reset()
    env.unwrapped.room_fixed = info['room_fixed']
    env.unwrapped.room_state = info['room_state']
    env.unwrapped.box_mapping = info['box_mapping']
    env.unwrapped.player_position = np.argwhere(env.room_state == 5)[0]

    return env
