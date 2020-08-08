import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from constants import Action


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        self.x_pre = 0
        self.y_pre = 0
        self.energy_pre = 0
        self.score_pre = self.state.score#Storing the last score for designing the reward function

    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, map_id, pos_x, pos_y, init_energy=50, max_steps=100):#tell server which map to run
        request = 'map%d,%d,%d,%d,%d' % (map_id, pos_x, pos_y, init_energy, max_steps)
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.x_pre = self.state.x
        self.y_pre = self.state.y
        self.energy_pre = self.state.energy
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1,
                         self.state.mapInfo.max_y + 1], dtype=int)
        for obstacle in self.state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx']
            y = obstacle['posy']
            view[x, y] = -obstacle_type

        for gold in self.state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y] = gold_amount

        players = [{
            'id': self.state.id,
            'x': self.state.x,
            'y': self.state.y,
            'energy': self.state.energy,
            'score': self.state.score,
        }]
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                players.append({
                    'id': player["playerId"],
                    'x': player["posx"],
                    'y': player["posy"],
                })

        return view, players

    def get_readable_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1,
                         self.state.mapInfo.max_y + 1], dtype=int)
        for obstacle in self.state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx']
            y = obstacle['posy']
            view[x, y] = -obstacle_type

        for gold in self.state.mapInfo.golds:
            gold_amount = gold['amount']
            x = gold['posx']
            y = gold['posy']
            if gold_amount > 0:
                view[x, y] = gold_amount

        # Add position and energy of agent to the DQNState
        playerStates = [{
            'id': self.state.id,
            'x': self.state.x,
            'y': self.state.y,
            'energy': self.state.energy,
            'score': self.state.score,
        }]
        #Add position of bots
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                playerStates.append({
                    'id': player['playerId'],
                    'x': player['posx'],
                    'y': player['posy'],
                })

        return {
            'map': view.tolist(),
            'players': playerStates,
        }

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action / 50
            
        if self.state.energy >= 45 and Action(self.state.lastAction) == Action.FREE:
            reward += -0.2

        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -1.0

        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -1.0

        # If there is no gold, but the agent still crafts golds, it will be punished.
        if Action(self.state.lastAction) == Action.CRAFT and self.energy_pre - self.state.energy == 10:
            reward += -0.2

        # If the agent is standing on a rich gold mine and its energy is enough but it didn't craft then it will be punished. 
        if Action(self.state.lastAction) != Action.CRAFT and self.state.mapInfo.gold_amount(self.x_pre, self.y_pre) >= 50 and self.energy_pre > 15:
            reward += -0.2
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
