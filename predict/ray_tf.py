import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog
import numpy as np


from MultiAgentGymMinerEnv import MultiAgentGymMinerEnv
from model.ray_tf_model import RayTFModel


class RayTFPredictor():
    def __init__(self):
        ModelCatalog.register_custom_model("my_model", RayTFModel)
        ray.init(dashboard_host='0.0.0.0')
        single_env = MultiAgentGymMinerEnv({})
        self.config = apex.APEX_DEFAULT_CONFIG.copy()
        self.config['num_workers'] = 0
        self.config['model']['custom_model'] = 'my_model'
        self.config['exploration_config'] = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.0,
            "final_epsilon": 0.0,
            "epsilon_timesteps": 1,
        }
        self.config['multiagent'] = {
            'policies': {
                'dqn': (None, single_env.observation_space, single_env.action_space, {
                    'model': {
                        'custom_model': 'my_model'
                    },
                    'exploration_config': {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 0.0,
                        "final_epsilon": 0.0,
                        "epsilon_timesteps": 1,
                    }
                })
            },
            'policy_mapping_fn': lambda agent_id: 'dqn',
        }
        self.agent = apex.ApexTrainer(env=MultiAgentGymMinerEnv, config=self.config)
        self.agent.restore('TrainedModels/ray_tf_v1/2020_08_22/checkpoint-721')

    def compute_action(self, state):
        state = self.preprocess(state)
        return self.agent.compute_action(state, policy_id='dqn')
    
    def preprocess(self, state):
        # Building the map
        view = np.zeros([21, 9, 6], dtype=float)
        for obstacle in state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx']
            y = obstacle['posy']
            if obstacle_type != 0:
                obstacle_type += 1
            view[x, y, 0] = obstacle_type

        for gold in state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y, 0] = 1
                view[x, y, 1] = gold_amount / 1000

        energies = np.zeros(4)
        i = 3
        for player in state.players:
            x = player['posx']
            y = player['posy']
            if x < view.shape[0] and y < view.shape[1]:
                if player['playerId'] == 1:
                    view[x, y, 2] = 1
                    energies[0] = player['energy']
                else:
                    view[x, y, i] = 1
                    energies[i - 2] = player['energy']
                    i += 1

        return (
            view,
            energies
        )
