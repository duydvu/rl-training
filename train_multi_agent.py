import random
import ray
from ray.rllib.agents.dqn import apex
from ray.rllib.models import ModelCatalog
from mlflow import log_metric, log_param, log_artifacts

from src.envs.multi_steps_multi_agents_env import MultiStepsMultiAgentsEnv as env
from src.models.ray_tf_model_v2 import RayTFModel
from configs.v1 import config


if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_model", RayTFModel)

    ray.init()

    agent = apex.ApexTrainer(env=env, config=config)

    for n in range(1000):
        result = agent.train()
        print(f'Step {n} - episode_reward_mean: {result["episode_reward_mean"]}')
        print(result['policy_reward_mean'])
        if (n + 1) % 20 == 0:
            agent.save('models/v1')

    ray.shutdown()
