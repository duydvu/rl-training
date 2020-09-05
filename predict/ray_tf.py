import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog
import numpy as np
import math


from MultiAgentGymMinerEnv import MultiAgentGymMinerEnv
from model.ray_tf_model import RayTFModel
from policy.random_policy import RandomPolicy


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
                        'custom_model': 'my_model',
                    },
                    'gamma': 0.99,
                }),
                'random': (RandomPolicy, obs_space, act_space, {}),
            },
            'policy_mapping_fn': lambda agent_id: 'dqn1',
        }
        self.agent = apex.ApexTrainer(env=MultiAgentGymMinerEnv, config=self.config)
        self.agent.restore('checkpoints/checkpoint_1240/checkpoint-1240')

    def reset(self):
        pass

    def compute_action(self, state):
        state = self.preprocess(state)
        return self.agent.compute_action(state, policy_id='dqn1', explore=False)
    
    def preprocess(self, state):
        # Building the map
        view = np.zeros([21, 9, 1], dtype=float)
        for obstacle in state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx']
            y = obstacle['posy']
            value = obstacle['value']
            if obstacle_type == 3:
                if value == -5:
                    obstacle_type = 4
                elif value == -20:
                    obstacle_type = 5
                elif value == -40:
                    obstacle_type = 6
                elif value == -100:
                    obstacle_type = 7
                else:
                    raise Exception('No such obstacle')
            view[x, y, 0] = obstacle_type

        for gold in state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y, 0] = min(7 + math.ceil(gold_amount / 50), 37)

        players_pos = np.full(4, -1, dtype=int)
        energies = np.zeros(4)
        i = 1
        for player_id, player_state in state.players.items():
            x = player_state['posx']
            y = player_state['posy']
            if x < view.shape[0] and y < view.shape[1]:
                if player_id == 1:
                    players_pos[0] = x * 9 + y
                    energies[0] = player_state['energy'] / 50
                else:
                    players_pos[i] = x * 9 + y
                    energies[i] = player_state['energy'] / 50
                    i += 1

        return (
            view,
            players_pos,
            energies,
        )
