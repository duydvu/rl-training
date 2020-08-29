import random
import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog

from MultiAgentGymMinerEnv import MultiAgentGymMinerEnv
from model.ray_tf_model import RayTFModel
from policy.random_policy import RandomPolicy
from recorder import Recorder

ModelCatalog.register_custom_model("my_model", RayTFModel)

ray.init()

single_env = MultiAgentGymMinerEnv({})
obs_space = single_env.observation_space
act_space = single_env.action_space

config = apex.APEX_DEFAULT_CONFIG.copy()
config['num_workers'] = 8
config['num_envs_per_worker'] = 4
config['num_gpus'] = 1
config['multiagent'] = {
    'policies': {
        'dqn1': (None, obs_space, act_space, {
            'model': {
                'custom_model': 'my_model'
            },
            'gamma': 0.99
        }),
        'dqn2': (None, obs_space, act_space, {
            'model': {
                'custom_model': 'my_model'
            },
            'gamma': 1.0
        }),
        'dqn3': (None, obs_space, act_space, {
            'model': {
                'custom_model': 'my_model'
            },
            'gamma': 0.1
        }),
        'random': (RandomPolicy, obs_space, act_space, {}),
    },
    'policy_mapping_fn': lambda agent_id: 'dqn1' if agent_id == '1' else random.choice(['dqn1', 'dqn2', 'dqn3', 'random']),
    'policies_to_train': ['dqn1', 'dqn2', 'dqn3']
}
config['buffer_size'] = 500000

agent = apex.ApexTrainer(env=MultiAgentGymMinerEnv, config=config)

recorder = Recorder(['Step', 'episode_reward_mean'])

for n in range(1000):
    result = agent.train()
    print(f'Step {n} - episode_reward_mean: {result["episode_reward_mean"]}')
    print(result['policy_reward_mean'])
    recorder.append([n, result["episode_reward_mean"]])
    if n % 20 == 0:
        agent.save('checkpoints')

ray.shutdown()
