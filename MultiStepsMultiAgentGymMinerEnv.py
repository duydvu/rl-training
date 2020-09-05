import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Tuple, Box, Dict, Discrete
from ray.rllib.utils.spaces.repeated import Repeated
import random
import math

from MinerEnv import MinerEnv
from constants import Action
from policy.bot1 import Bot1
from policy.bot2 import Bot2


class MultiStepsMultiAgentGymMinerEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = MinerEnv(None, None)
        self.env.start()
        self.state = self.env.state
        self.width = 21
        self.height = 9
        self.max_step = env_config['max_step']
        self.action_space = Discrete(6)
        self.observation_space = Tuple((
            Box(low=0, high=np.inf, shape=(self.width, self.height)),
            Box(low=-np.inf, high=np.inf, shape=(self.max_step, 4)),
            Box(low=-2, high=1, shape=(4,)),
        ))
        self.step_states = []

    def reset(self):
        map_id = np.random.randint(1, 7)
        pos_x = np.random.randint(self.width)
        pos_y = np.random.randint(self.height)
        number_of_players = np.random.randint(1, 5)
        self.env.send_map_info(map_id, pos_x, pos_y,
                               number_of_players=number_of_players)
        self.env.reset()
        ids = list(range(2, 1 + number_of_players))
        self.bots = []
        if number_of_players > 1:
            for _ in range(np.random.randint(1, number_of_players)):
                if random.choice([1, 1, 2]) == 1:
                    self.bots.append(Bot1(ids.pop(random.choice(range(len(ids))))))
                else:
                    self.bots.append(Bot2(ids.pop(random.choice(range(len(ids)))), gamma=random.choice([0.9, 0.95, 1.0])))
        step_state = self.get_state()
        self.step_states = {
            player_id_str: (
                player_state[0],
                [player_state[1] for _ in range(self.max_step)],
                player_state[2],
            )
            for player_id_str, player_state in step_state.items()
        }
        return self.step_states

    def step(self, action):
        for bot in self.bots:
            action[str(bot.id)] = bot.compute_action(self.state)
        self.env.step(action)
        step_state = self.get_state()
        for player_id_str, player_state in step_state.items():
            self.step_states[player_id_str] = (
                step_state[player_id_str][0],
                self.step_states[player_id_str][1][1:] + [step_state[player_id_str][1]],
                step_state[player_id_str][2],
            )
        return self.step_states, self.get_reward(), self.get_done(), {}

    def get_state(self):
        # Building the map
        view = np.zeros([self.width, self.height], dtype=float)
        for obstacle in self.state.mapInfo.obstacles:
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
            view[x, y] = obstacle_type

        for gold in self.state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y] = min(7 + math.ceil(gold_amount / 50), 37)

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

        if player['status'] == self.state.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -6.0

        if player['status'] == self.state.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -6.0

        if Action(player['lastAction']) == Action.FREE and player_pre['energy'] == 50:
            reward += -0.1

        return reward
    
    def get_done(self):
        done = {'__all__': False}
        if all(map(lambda player_state: player_state['status'] != 0, self.state.players.values())):
            done['__all__'] = True
        return done
