import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog
import numpy as np


from MultiAgentGymMinerEnv import MultiAgentGymMinerEnv
from model.ray_tf_model import RayTFModel


class RayTFPredictor():
    def __init__(self):
        ModelCatalog.register_custom_model("my_model", RayTFModel)
        ray.init()
        single_env = MultiAgentGymMinerEnv({})
        obs_space = single_env.observation_space
        act_space = single_env.action_space
        self.config = apex.APEX_DEFAULT_CONFIG.copy()
        self.config['num_workers'] = 0
        self.config['in_evaluation'] = True
        self.config['multiagent'] = {
            'policies': {
                'dqn1': (None, obs_space, act_space, {
                    'model': {
                        'custom_model': 'my_model'
                    },
                    'gamma': 0.99
                })
            },
            'policy_mapping_fn': lambda agent_id: 'dqn1',
        }
        self.agent = apex.ApexTrainer(env=MultiAgentGymMinerEnv, config=self.config)
        self.agent.restore('checkpoints/checkpoint_81/checkpoint-81')

    def compute_action(self, state):
        state = self.preprocess(state)
        return self.agent.compute_action(state, policy_id='dqn1', explore=False)
    
    def preprocess(self, state):
        # Building the map
        view = np.zeros([21, 9, 6], dtype=float)
        for obstacle in state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx']
            y = obstacle['posy']
            value = obstacle['value']
            if obstacle_type != 0:
                obstacle_type += 1
            if obstacle_type == 4:
                if value == -5:
                    obstacle_type = 5
                elif value == -20:
                    obstacle_type = 6
                elif value == -40:
                    obstacle_type = 7
                elif value == -100:
                    obstacle_type = 8
                else:
                    raise Exception('No such obstacle')
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
        for player_id, player_state in state.players.items():
            x = player_state['posx']
            y = player_state['posy']
            if x < view.shape[0] and y < view.shape[1]:
                if player_id == 1:
                    view[x, y, 2] = 1
                    energies[0] = player_state['energy'] / 50
                else:
                    view[x, y, i] = 1
                    energies[i - 2] = player_state['energy'] / 50
                    i += 1

        return (
            view,
            energies
        )
