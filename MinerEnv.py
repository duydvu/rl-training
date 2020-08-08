import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from constants import Action


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
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
        view = np.zeros([self.state.mapInfo.max_x + 3, self.state.mapInfo.max_y + 3, 5], dtype=float)
        for obstacle in self.state.mapInfo.obstacles:
            obstacle_type = obstacle['type']
            x = obstacle['posx'] + 1
            y = obstacle['posy'] + 1
            view[x, y, 0] = -obstacle_type

        for gold in self.state.mapInfo.golds:
            gold_amount = gold['amount'] / 1000
            x = gold['posx'] + 1
            y = gold['posy'] + 1
            if gold_amount > 0:
                view[x, y, 0] = gold_amount

        view[self.state.x + 1, self.state.y + 1, 1] = 1
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                x = player["posx"] + 1
                y = player["posy"] + 1
                view[x, y, player['playerId']] = 1

        return [view, [self.state.energy / 50]]

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
            
        if self.state.energy >= 45 and self.state.lastAction == Action.REST:
            reward += -0.2

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -0.2
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -0.2
        # print ("reward",reward)
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
