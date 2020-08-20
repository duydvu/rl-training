import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog

from model.ray_tf_model import RayTFModel
from GymMinerEnv import GymMinerEnv



class RayTFPredictor():
    def __init__(self):
        ModelCatalog.register_custom_model("my_model", RayTFModel)
        ray.init(dashboard_host='0.0.0.0')
        self.config = apex.APEX_DEFAULT_CONFIG.copy()
        self.config['num_workers'] = 0
        self.config['model']['custom_model'] = 'my_model'
        self.config['exploration_config'] = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.0,
            "final_epsilon": 0.0,
            "epsilon_timesteps": 1,
        }
        self.agent = apex.ApexTrainer(env=GymMinerEnv, config=self.config)
        self.agent.restore('TrainedModels/ray_tf_v1/checkpoints/checkpoint_441/checkpoint-441')

    def compute_action(self, state):
        return self.agent.compute_action(state)
