import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog
import numpy as np


from MultiStepsMultiAgentGymMinerEnv import MultiStepsMultiAgentGymMinerEnv
from model.ray_tf_model_v2 import RayTFModel


class RayTFPredictor():
    def __init__(self):
        ModelCatalog.register_custom_model("my_model", RayTFModel)
        ray.init()
        self.max_step = 5
        single_env = MultiStepsMultiAgentGymMinerEnv({'max_step': self.max_step})
        obs_space = single_env.observation_space
        act_space = single_env.action_space
        self.config = apex.APEX_DEFAULT_CONFIG.copy()
        self.config = {
            **self.config,
            'num_workers': 0,
            'in_evaluation': True,
            'multiagent': {
                'policies': {
                    'dqn1': (None, obs_space, act_space, {
                        'model': {
                            'custom_model': 'my_model',
                            'custom_model_config': {
                                'max_step': 5,
                                'embedding_size': 8,
                                'conv1_filters': [
                                    (32, 4, 1),
                                    (64, 2, 2),
                                    (128, 2, 2),
                                ],
                                'conv2_filter': (128, 2, 1)
                            }
                        },
                        'gamma': 0.99
                    })
                },
                'policy_mapping_fn': lambda agent_id: 'dqn1',
            },
            'env_config': {
                'max_step': self.max_step,
            }
        }
        self.agent = apex.ApexTrainer(
            env=MultiStepsMultiAgentGymMinerEnv, config=self.config)
        self.agent.restore('checkpoints/checkpoint_881/checkpoint-881')
        self.step_states = []
    
    def reset(self):
        self.step_states = []

    def compute_action(self, state):
        step_state = self.preprocess(state)
        if len(self.step_states) == 0:
            self.step_states = (
                [step_state[0] for _ in range(self.max_step)],
                [step_state[1] for _ in range(self.max_step)],
            )
        else:
            self.step_states = (
                self.step_states[0][1:] + [step_state[0]],
                self.step_states[1][1:] + [step_state[1]],
            )
        return self.agent.compute_action(self.step_states, policy_id='dqn1', explore=False)

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
