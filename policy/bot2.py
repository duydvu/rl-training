from MINER_STATE import State
import numpy as np
import random


class Bot2:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, playerId=1, gamma=0.95):
        self.id = playerId
        self.gamma = gamma
        self.reset()
    
    def reset(self):
        self.target_gold = [-1, -1]

    def compute_action(self, state: State):
        player = state.players[self.id]
        x = player['posx']
        y = player['posy']
        energy = player['energy']

        if state.mapInfo.gold_amount(x, y) > 0:
            if energy >= 6:
                return self.ACTION_CRAFT
            else:
                return self.ACTION_FREE

        if energy < 5:
            return self.ACTION_FREE

        if state.mapInfo.gold_amount(*self.target_gold) == 0:
            max_amount = 0
            for cell in state.mapInfo.golds:
                distance = abs(cell['posx'] - x) + abs(cell['posy'] - y)
                amount = cell['amount'] * (self.gamma ** distance)
                if amount > max_amount:
                    max_amount = amount
                    self.target_gold = [cell['posx'], cell['posy']]


        if random.choice([0, 1]) == 0:
            if self.target_gold[0] - x > 0:
                right_obs = state.mapInfo.get_obstacle(x + 1, y)
                return self.go_forward(state, energy, right_obs, self.ACTION_GO_RIGHT, [self.ACTION_GO_DOWN, self.ACTION_GO_UP])
            elif self.target_gold[0] - x < 0:
                left_obs = state.mapInfo.get_obstacle(x - 1, y)
                return self.go_forward(state, energy, left_obs, self.ACTION_GO_LEFT, [self.ACTION_GO_DOWN, self.ACTION_GO_UP])

            if self.target_gold[1] - y > 0:
                down_obs = state.mapInfo.get_obstacle(x, y + 1)
                return self.go_forward(state, energy, down_obs, self.ACTION_GO_DOWN, [self.ACTION_GO_LEFT, self.ACTION_GO_RIGHT])
            elif self.target_gold[1] - y < 0:
                up_obs = state.mapInfo.get_obstacle(x, y - 1)
                return self.go_forward(state, energy, up_obs, self.ACTION_GO_UP, [self.ACTION_GO_LEFT, self.ACTION_GO_RIGHT])
        else:
            if self.target_gold[1] - y > 0:
                down_obs = state.mapInfo.get_obstacle(x, y + 1)
                return self.go_forward(state, energy, down_obs, self.ACTION_GO_DOWN, [self.ACTION_GO_LEFT, self.ACTION_GO_RIGHT])
            elif self.target_gold[1] - y < 0:
                up_obs = state.mapInfo.get_obstacle(x, y - 1)
                return self.go_forward(state, energy, up_obs, self.ACTION_GO_UP, [self.ACTION_GO_LEFT, self.ACTION_GO_RIGHT])

            if self.target_gold[0] - x > 0:
                right_obs = state.mapInfo.get_obstacle(x + 1, y)
                return self.go_forward(state, energy, right_obs, self.ACTION_GO_RIGHT, [self.ACTION_GO_DOWN, self.ACTION_GO_UP])
            elif self.target_gold[0] - x < 0:
                left_obs = state.mapInfo.get_obstacle(x - 1, y)
                return self.go_forward(state, energy, left_obs, self.ACTION_GO_LEFT, [self.ACTION_GO_DOWN, self.ACTION_GO_UP])

        raise Exception('Not handled case.')

    
    def go_forward(self, state, energy, obs, main_action, alter_actions):
        if obs == -1 or obs['type'] == 0:
            return main_action
        elif obs['type'] == 1:
            if energy <= 20:
                return self.ACTION_FREE
            else:
                return main_action
        elif obs['type'] == 2:
            if energy <= 10:
                return self.ACTION_FREE
            else:
                return main_action
        elif obs['type'] == 3 and obs['value'] > -100:
            if energy <= -obs['value']:
                return self.ACTION_FREE
            else:
                return main_action
        elif obs['type'] == 3 and obs['value'] == -100:
            return random.choice(alter_actions)
