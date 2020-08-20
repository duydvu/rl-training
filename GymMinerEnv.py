import numpy as np
import gym
from gym.spaces import Tuple, Box, Dict, Discrete
from ray.rllib.utils.spaces.repeated import Repeated

from MinerEnv import MinerEnv


class GymMinerEnv(gym.Env):
    def __init__(self, env_config):
        self.env = MinerEnv(None, None)
        self.env.start()
        self.state = self.env.state
        self.width = 21
        self.height = 9
        self.action_space = Discrete(6)
        self.observation_space = Tuple((
            Box(low=0, high=np.inf, shape=(self.width, self.height, 5)),
            Box(low=-np.inf, high=np.inf, shape=(4,)),
            # Box(low=0, high=np.inf, shape=(self.width, self.height)),
            # Repeated(Dict({
            #     'position': Box(low=0, high=1, shape=(self.width, self.height)),
            #     'energy': Box(low=-np.inf, high=np.inf, shape=(1,))
            # }), max_len=4)
        ))

    def reset(self):
        map_id = np.random.randint(1, 6)
        pos_x = np.random.randint(self.width)
        pos_y = np.random.randint(self.height)
        self.env.send_map_info(map_id, pos_x, pos_y)
        self.env.reset()
        return self.get_state()

    def step(self, action):
        self.env.step(str(action))
        return self.get_state(), self.env.get_reward(), self.env.check_terminate(), {}

    def get_state(self):
        # Building the map
        view = np.zeros([self.width, self.height, 5], dtype=float)
        gold_amount_view = np.zeros([self.width, self.height], dtype=float)
        for obstacle in self.state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx']
            y = obstacle['posy']
            # if obstacle_type != 0:
            #     obstacle_type += 1
            view[x, y, 0] = obstacle_type

        for gold in self.state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y, 0] = gold_amount / 1000
                # gold_amount_view[x, y] = gold_amount / 1000

        players = []
        for player in self.state.players:
            # player_pos_onehot = np.zeros_like(view)
            # player_pos_onehot[player['posx'], player['posy']] = 1
            # players.append({
            #     'position': player_pos_onehot,
            #     'energy': [player['energy']]
            # })
            x = player['posx']
            y = player['posy']
            if x < view.shape[0] and y < view.shape[1]:
                view[x, y, player['playerId']] = 1
            players.append(player['energy'])

        return (
            view,
            # gold_amount_view,
            players
        )

if __name__ == '__main__':
    env = GymMinerEnv({})
    env.reset()
    print(env.get_state())
