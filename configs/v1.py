from ray.rllib.agents.dqn import apex

from src.envs.multi_steps_multi_agents_env import MultiStepsMultiAgentsEnv as env
from src.policies.random_policy import RandomPolicy

max_step = 3
single_env = env({'max_step': max_step})
obs_space = single_env.observation_space
act_space = single_env.action_space

config = apex.APEX_DEFAULT_CONFIG.copy()
config = {
    **config,
    'num_workers': 8,
    'num_envs_per_worker': 4,
    'num_gpus': 1,
    'buffer_size': 1000000,
    'env_config': {
        'max_step': max_step,
    },
    'multiagent': {
        'policies': {
            'dqn1': (None, obs_space, act_space, {
                'model': {
                    'custom_model': 'my_model',
                    'custom_model_config': {
                        'max_step': max_step,
                        'embedding_size': 8,
                        'conv1_filters': [
                            (32, 4, 1),
                            (64, 2, 2),
                            (128, 2, 2),
                        ]
                    }
                },
                'gamma': 0.99,
            }),
            'random': (RandomPolicy, obs_space, act_space, {}),
        },
        'policy_mapping_fn': lambda agent_id: 'dqn1' if agent_id == '1' else 'random',
        'policies_to_train': ['dqn1']
    }
}
