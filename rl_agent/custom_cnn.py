import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomGridCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] #which is 9 in our case
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64*7*7, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))
policy_kwargs = dict(
    features_extractor_class = CustomGridCNN,
    features_extractor_kwargs = dict(features_dim=128)
)

