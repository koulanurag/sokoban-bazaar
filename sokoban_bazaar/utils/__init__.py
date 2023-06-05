import os
from pathlib import Path
def domain_pddl_path(env_name):
    assets_dir = os.path.join(Path(os.path.dirname(__file__)).parent, 'assets')
    return os.path.join(assets_dir, 'sokoban_domain.pddl')