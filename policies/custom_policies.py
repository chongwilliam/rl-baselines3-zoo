from stable_baselines3.common.policies import ActorCriticPolicy
from models.tcn_feature_extractor import TCNFeatureExtractor

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=TCNFeatureExtractor,
            features_extractor_kwargs={"features_dim": 128},
        )