import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Tuple, Box, Dict, Discrete
from ray.rllib.utils.spaces.repeated import Repeated

from MinerEnv import MinerEnv
from constants import Action


class MultiAgentGymMinerEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = MinerEnv(None, None)
        self.env.start()
        self.state = self.env.state
        self.width = 21
        self.height = 9
        self.action_space = Discrete(6)
        self.observation_space = Tuple((
            Box(low=0, high=np.inf, shape=(self.width, self.height, 2)),
            Box(low=-np.inf, high=np.inf, shape=(4,)),
            Box(low=-2, high=1, shape=(4,)),
        ))

    def reset(self):
        map_id = np.random.randint(1, 7)
        pos_x = np.random.randint(self.width)
        pos_y = np.random.randint(self.height)
        number_of_players = np.random.randint(1, 5)
        self.env.send_map_info(map_id, pos_x, pos_y,
                               number_of_players=number_of_players)
        self.env.reset()
        return self.get_state()

    def step(self, action):
        self.env.step(action)
        return self.get_state(), self.get_reward(), self.get_done(), {}

    def get_state(self):
        # Building the map
        view = np.zeros([self.width, self.height, 2], dtype=float)
        for obstacle in self.state.mapInfo.obstacles:
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

        for gold in self.state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y, 0] = 1
                view[x, y, 1] = gold_amount / 1000

        return {
            str(player_id): self.get_single_player_state(np.copy(view), player_id)
            for player_id in self.state.players.keys()
        }

    def get_single_player_state(self, view, playerId):
        players_pos = np.full(4, -1, dtype=int)
        energies = np.zeros(4)
        i = 1
        for player_id, player_state in self.state.players.items():
            x = player_state['posx']
            y = player_state['posy']
            if x < view.shape[0] and y < view.shape[1]:
                if player_id == playerId:
                    players_pos[0] = x * self.height + y
                    energies[0] = player_state['energy'] / 50
                else:
                    players_pos[i] = x * self.height + y
                    energies[i] = player_state['energy'] / 50
                    i += 1

        return (
            view,
            players_pos,
            energies,
        )

    def get_reward(self):
        return {
            str(player_id): self.get_single_player_reward(player_id)
            for player_id in self.state.players.keys()
        }
    
    def get_single_player_reward(self, playerId):
        # Calculate reward
        reward = 0
        player = self.state.players[playerId]
        player_pre = self.state.players_pre[playerId]
        score_action = player['score'] - player_pre['score']
        if score_action > 0:
            reward += score_action / 50

        consumed_energy = player_pre['energy'] - player['energy']
        if Action(self.state.players[playerId]['lastAction']) == Action.CRAFT and consumed_energy == 10:
            reward += -1.0

        return reward
    
    def get_done(self):
        done = {'__all__': False}
        if all(map(lambda player_state: player_state['status'] != 0, self.state.players.values())):
            done['__all__'] = True
        return done
