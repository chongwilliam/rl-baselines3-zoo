from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register, register_envs

from rl_zoo3.wrappers import MaskVelocityWrapper

# Custom import from envs folder 
import sys 
import os 
package_path = os.path.join(os.path.dirname(__file__), '../envs')
sys.path.insert(0, package_path)

# Add custom pid environment
try:
    print("Imported force environment")

    register(
        id='force_env-v0',          # Unique ID for your environment
        entry_point='force_env:SecondOrderPIDControlEnv',  # Path to your environment
    )

    env = gym.make('force_env-v0')

except ImportError:
    print("Force environment unable to be imported")

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

try:
    import ale_py

    # no-op
    gym.register_envs(ale_py)
except ImportError:
    pass

try:
    import highway_env
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs
except ImportError:
    pass

try:
    import gym_donkeycar
except ImportError:
    pass

try:
    import panda_gym
except ImportError:
    pass

try:
    import rocket_lander_gym
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass


# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )
