import torch as th 
import torch.nn as nn
from pytorch_tcn import TCN  # Import TCN from pytorch_tcn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np 

class TCNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, num_channels=[32, 64, 128]):
        
        # Observation space is (channels, data)         
        # Initialize with the final feature dimension, which is the output channel count of the last TCN layer
        super(TCNFeatureExtractor, self).__init__(observation_space, features_dim=num_channels[-1])
        
        n_input_channels = observation_space.shape[0]  # Number of features per time step
        self.tcn = nn.Sequential(TCN(n_input_channels, num_channels), nn.Flatten())

        # Compute shape by doing one forward pass
        # print(th.as_tensor(observation_space.sample()[None]).float().shape[:])
        with th.no_grad():
            n_flatten = self.tcn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
                
        # hard-code, since output of TCN is the same size as the last TCN layer 
        self.linear = nn.Sequential(nn.Linear(n_flatten, num_channels[-1]), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.tcn(observations))

if __name__ == "__main__":
    print("TCN Test")
    
    window_size = 100
    obs_space = spaces.Box(low=-1, high=1, shape=(1, 100), dtype=np.float32)
    tcn_network = TCNFeatureExtractor(obs_space)
    
    # tcn input is (N, Cin, L) : N: batch size, Cin: number of input channels, L: sequence length 
    obs_sample = th.as_tensor(obs_space.sample()[None]).float()
    print(obs_sample.shape[:])
    print(tcn_network.forward(obs_sample).shape[:])  # last output dimension size (1, 128)
    