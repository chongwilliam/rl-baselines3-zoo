import gymnasium as gym 
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
import numpy as np
import random 

from utils.tcn_feature_extractor import TCNFeatureExtractor
from utils.noise import WhiteNoise
from utils.low_pass_filter import LowPassFilter
from utils.compute_poles import compute_optimal_gains, solve_fast_optimization

class SecondOrderPIDControlEnv(gym.Env):
    def __init__(self, 
                 delay_steps=1, 
                 wn=5, 
                 kv=10, 
                 Ks=1000.0, 
                 Ks_env=100.0, 
                 Kp=0.1, 
                 Ki=1.0, 
                 Kd=0, 
                 max_steps=1e9, 
                 timestep=0.001, 
                 window_size=100, 
                 Ks_env_min=10,
                 Ks_env_max=10000,
                 Ks_min=10, 
                 Ks_max=5000,
                 action_scaling = 10000./1000):
        super().__init__()
        
        # Define action and observation spaces
        # self.action_space = spaces.Box(low=10, high=1000.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # action space is the derivative of the stiffness
        # self.action_space = spaces.Box(low=np.array([0., 0., 1.]), high=np.array([0.5, 2.5, 10.]), shape=(3,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-10, high=10, shape=(1, window_size), dtype=np.float32)  # normalization of input data based on error / target force 
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, window_size), dtype=np.float32)  # normalization of input data (error / target_force)
        
        # Target output we want the system to reach
        self.target = 0
        
        # System parameters
        self.wn = wn  # Desired natural frequency of the critically-damped response 
        self.kv = kv  # Damping coefficient
        self.Ks_action = Ks  # Estimated environment stiffness (action output)
        self.delay_steps = delay_steps  # Time delay in steps
        self.max_steps = max_steps
        self.timestep = timestep  # Discrete integration timestep (0.001 seconds)
        self.window_size = window_size
        
        # Environment parameters 
        self.Ks_env = Ks_env  # Environment stiffness (actual stiffness)
        self.Ks_min = Ks_min 
        self.Ks_max = Ks_max
        self.Ks_env_min = Ks_env_min
        self.Ks_env_max = Ks_env_max
        self.action_scaling = action_scaling

        # LPF filter for output of policy
        self.lpf = LowPassFilter(cutoff=10, fs=1000)
        
        # PID gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd 

        # System state variables
        self.position = 0.0  # Position (y)
        self.velocity = 0.0  # Velocity (dy/dt)
        self.integral_error = 0.0  # Integral of the error
        self.previous_error = 0.0  # Previous error for derivative calculation
        self.current_step = 0
        
        # Queue to store delayed actions
        self.action_queue = [0.0] * delay_steps
        self.first_loop = True 
        self.prev_action = 0
        self.alpha = 100.0
        self.reward_deadband = 0.01
        
        # White noise generator for force sensor to add to self.position for error signal
        self.white_noise = WhiteNoise(5e-2)  # from adev generation 

        # Buffer for observation space (?)
        self.obs_buffer = [0.0] * window_size

    def compute_pid_action(self, error):
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self.integral_error += error * self.timestep
        I = self.Ki * self.integral_error
        
        # Derivative term
        D = self.Kd * (error - self.previous_error) / self.timestep
        self.previous_error = error
        
        # PI output
        pid_action = P + I 
        return pid_action
    
    def set_target(self, target):
        self.target = target 
        
    def set_env_stiffness(self, Ks_env):
        self.Ks_env = Ks_env

    def set_wn(self, wn):
        self.wn = wn 

    # Define the RK4 step function
    def rk4_step(self, dynamics, y, t, h, u_prev, kv):
        """Perform one RK4 step with a delayed input."""
        k1 = dynamics(t, y, u_prev, kv)
        k2 = dynamics(t + h / 2, y + h / 2 * k1, u_prev, kv)
        k3 = dynamics(t + h / 2, y + h / 2 * k2, u_prev, kv)
        k4 = dynamics(t + h, y + h * k3, u_prev, kv)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Define the system dynamics
    def dynamics(self, t, y, u, kv):
        """Compute derivatives for the system."""
        # kv = 1.0  # Example value for kv
        y1, y2 = y
        dy1 = y2
        dy2 = float(u - kv * y2)
        return np.array([dy1, dy2])
    
    def reward(self, error):
        """Compute reward"""
        # if abs(error) < self.reward_deadband:
        #     return 2 - abs(error) - error**2
        # else:
        #     return - error**2
        return -float(error)**2
            
    def step(self, action):

        # If first loop, push action to previous action 
        if self.first_loop:
            self.prev_action = action 
            self.first_loop = False 

        # Get filtered action and saturate within bounds 
        filtered_action = action 
        # filtered_action = self.lpf.update(action)
        # filtered_action = max(10, min(filtered_action, 1000))
        
        # Integrate to get stiffness estimate
        self.Ks_action += filtered_action * self.action_scaling 
        if (self.Ks_action < self.Ks_min):
            self.Ks_action = self.Ks_min
        elif (self.Ks_action > self.Ks_max):
            self.Ks_action = self.Ks_max 

        # Given current estimate of the environment stiffness, compute the optimal PID gains 
        # action = self.Ks_env
        # self.Kp, self.Ki, self.kv = compute_optimal_gains(action, self.wn)  
        # self.Kp, self.Ki, self.kv = action 
        self.Kp, self.Ki, self.kv = solve_fast_optimization(kv_min=0.1, kv_max=10.0, kp_min=1e-3, kp_max=0.7, ki_min=1e-3, ki_max=2.5, ke=self.Ks_action, wn_upper_bound=10.0)
                
        # Calculate error
        # error = self.target - (self.Ks_env * self.position + self.white_noise.generate(1))
        sensed_force = self.Ks_env * self.position + self.white_noise.generate(1)
        error = sensed_force - self.target  # f_s - f_d

        # Compute the action using PID control based on the error
        pid_action = -self.compute_pid_action(error) 

        # Queue the current PID action to be applied after delay_steps
        self.action_queue.append(pid_action)
        delayed_action = self.action_queue.pop(0)  # Pop the oldest action (delayed)

        # Compute dynamics update with RK4 and unpack 
        y = np.array([self.position, self.velocity])
        y = self.rk4_step(self.dynamics, y, 0, self.timestep, delayed_action, self.kv)
        self.position = float(y[0])
        self.velocity = float(y[1])
                
        # Observation is the error based on the noisy position
        sensed_force = self.Ks_env * self.position + self.white_noise.generate(1)
        error = sensed_force - self.target 
        error_ratio = min(-1.0, max(error / self.target, 1.0))

        # observation = np.array([sensed_force - self.target], dtype=np.float32)  # not normalized with target force 
        # observation = np.array([(sensed_force - self.target) / self.target], dtype=np.float32)  # normalized with target force 
        observation = np.array([error_ratio], dtype=np.float32)
        self.obs_buffer.append(float(observation[0]))
        self.obs_buffer.pop(0)

        # Package observation buffer vector for TCN feature extraction 
        obs_buffer_vec = np.array(self.obs_buffer, dtype=np.float32).reshape(1, 100)
                
        # Reward is based on minimizing the absolute error and minimizing change in stiffness 
        # reward = -abs(observation[0]) - 0 * self.alpha * abs((action - self.prev_action))
        reward = self.reward(error)

        # Episode ends if max steps reached or error is within an acceptable range
        self.current_step += 1
        # done = self.current_step >= self.max_steps or abs(observation[0]) < 0.01
        done = False 

        # Step log
        # print("Error: ", error)
        # print("Target: ", self.target)
        # print("Action (K):", action)
        # print("Filtered action:", filtered_action)
        # print("Action stiffness:", self.Ks_action)
        # print("Actual stiffness:", self.Ks_env)
        # print("Gains (kp, ki, kv):", self.Kp, self.Ki, self.kv)
        
        # return observation, reward, done, False, {}
        return obs_buffer_vec, reward, done, False, {}

    def reset(self, seed=None, options=None):
        # Reset state
        self.position = 0.0
        self.velocity = 0.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.current_step = 0
        self.action_queue = [0.0] * self.delay_steps  # Reinitialize delay queue
        self.obs_buffer = [0.0] * self.window_size
        self.first_loop = True 
        self.lpf.reset()

        # Randomize target between values 
        self.target = random.uniform(0, 20.0)

        # Randomize environment stiffness
        self.Ks_env = random.uniform(self.Ks_env_min, self.Ks_env_max)
        
        return np.array(self.obs_buffer, dtype=np.float32).reshape(1, 100), {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step},  Position (y): {self.position}, Target: {self.target}, Error: {self.target - self.Ks_env * self.position}")

    def close(self):
        pass

# Check environment validity
if __name__ == "__main__":
    env = SecondOrderPIDControlEnv()
    check_env(env)