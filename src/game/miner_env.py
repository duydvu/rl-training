import sys
import json
import numpy as np

from .game_socket_dummy import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from .miner_state import State
from src.utils.constants import Action


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, map_id, pos_x, pos_y, init_energy=50, number_of_players=4, max_steps=100):#tell server which map to run
        request = 'map%d,%d,%d,%d,%d,%d' % (map_id, pos_x, pos_y, init_energy, number_of_players, max_steps)
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, actions_dict): #step process
        # send action to server
        self.socket.send(json.dumps(actions_dict, cls=NpEncoder))
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_state(self):
        return self.state

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

        playerStates = []
        for player_id, player_state in self.state.players.items():
            playerStates.append({
                'id': player_id,
                'x': player_state['posx'],
                'y': player_state['posy'],
                'energy': player_state['energy'],
                'score': player_state['score'],
            })

        return {
            'map': view.tolist(),
            'players': playerStates,
        }
