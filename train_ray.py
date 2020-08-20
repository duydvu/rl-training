import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog


from GymMinerEnv import GymMinerEnv
from model.ray_tf_model import RayTFModel
from recorder import Recorder

ModelCatalog.register_custom_model("my_model", RayTFModel)

ray.init(dashboard_host='0.0.0.0', dashboard_port=5202)

config = apex.APEX_DEFAULT_CONFIG.copy()
config['num_workers'] = 8
config['num_envs_per_worker'] = 4
config['num_gpus'] = 1

config['model']['custom_model'] = 'my_model'
agent = apex.ApexTrainer(env=GymMinerEnv, config=config)

recorder = Recorder(['Step', 'episode_reward_mean'])

for n in range(1000):
    result = agent.train()
    print(f'Step {n} - episode_reward_mean: {result["episode_reward_mean"]}')
    recorder.append([n, result["episode_reward_mean"]])
    if n % 20 == 0:
        agent.save('checkpoints')
