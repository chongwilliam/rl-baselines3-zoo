from gymnasium.envs.registration import register
import gymnasium as gym

# register(
    # id='force_env-v0',          # Unique ID for your environment
    # entry_point='force_env:SecondOrderPIDControlEnv',  # Path to your environment
# )

env = gym.make('force_env-v0')

