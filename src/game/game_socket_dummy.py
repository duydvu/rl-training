import sys
import logging
from array import *
import json
import os
import math
import copy
from random import randrange


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ObstacleInfo:
    # initial energy for obstacles: Land (key = 0): -1, Forest(key = -1): 0 (random), Trap(key = -2): -10, Swamp (key = -3): -5
    types = {
        0: -1,
        -1: 0,
        -2: -10,
        -3: -5,
    }

    def __init__(self, obs_type, posx, posy):
        self.type = -obs_type
        self.posx = posx
        self.posy = posy
        self.value = self.types[obs_type]


class GoldInfo:
    def __init__(self, posx, posy, amount):
        self.posx = posx
        self.posy = posy
        self.amount = amount


class PlayerInfo:
    STATUS_PLAYING = 0
    STATUS_ELIMINATED_WENT_OUT_MAP = 1
    STATUS_ELIMINATED_OUT_OF_ENERGY = 2
    STATUS_ELIMINATED_INVALID_ACTION = 3
    STATUS_STOP_EMPTY_GOLD = 4
    STATUS_STOP_END_STEP = 5

    def __init__(self, playerId):
        self.playerId = playerId
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = PlayerInfo.STATUS_PLAYING
        self.freeCount = 0


class GameInfo:
    def __init__(self, numberOfPlayers, steps):
        self.numberOfPlayers = numberOfPlayers
        self.steps = steps
        self.width = 0
        self.height = 0
        self.golds = []
        self.obstacles = []

    def load_map(self, map_obj):
        self.height = len(map_obj)
        self.width = len(map_obj[0])
        for i in range(self.height):
            for j in range(self.width):
                if map_obj[i][j] > 0:  # gold
                    g = GoldInfo(posx=j, posy=i, amount=map_obj[i][j])
                    self.golds.append(g)
                else:  # obstacles
                    o = ObstacleInfo(obs_type=map_obj[i][j], posx=j, posy=i)
                    self.obstacles.append(o)


class UserMatch:
    def __init__(self, posx, posy, energy, numberOfPlayers, steps):
        self.posx = posx
        self.posy = posy
        self.energy = energy
        self.gameinfo = GameInfo(numberOfPlayers, steps)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class StepState:
    def __init__(self):
        self.players = []
        self.golds = []
        self.changedObstacles = []

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class GameSocket:
    bog_energy_chain = {
        -5: -20,
        -20: -40,
        -40: -100,
        -100: -100,
    }

    def __init__(self, host, port):
        self.stepCount = 0
        self.maxStep = 0
        self.mapdir = "Maps"  # where to load all pre-defined maps
        self.users = []
        self.stepState = StepState()
        self.maps = {}  # key: map file name, value: file content
        self.map = []  # running map info: 0->Land, -1->Forest, -2->Trap, -3:Swamp, >0:Gold
        self.energyOnMap = []  # self.energyOnMap[x][y]: <0, amount of energy which player will consume if it move into (x,y)
        self.E = 0
        self.resetFlag = True
        self.craftUsers = []  # players that craft at current step - for calculating amount of gold
        self.craftMap = {}  # cells that players craft at current step, key: x_y, value: number of players that craft at (x,y)

        # load all pre-defined maps from mapDir
        for filename in os.listdir(self.mapdir):
            print("Found: " + filename)
            with open(os.path.join(self.mapdir, filename), 'r') as f:
                self.maps[filename] = json.loads(f.read())

    def reset(self, requests):  # load new game by given request: [map id (filename), posx, posy, initial energy]
        # load new map
        mapId, *requests = requests
        posx, posy, energy, numberOfPlayers, steps = list(map(int, requests))
        self.userMatch = UserMatch(
            posx=posx, posy=posy, energy=energy, numberOfPlayers=numberOfPlayers, steps=steps)
        self.reset_map(mapId)
        self.maxStep = self.userMatch.gameinfo.steps

        # init data for players
        self.users = []
        for user_id in range(1, numberOfPlayers + 1):
            user = PlayerInfo(user_id)
            user.posx = self.userMatch.posx
            user.posy = self.userMatch.posy
            user.energy = self.userMatch.energy
            user.lastAction = -1
            user.status = PlayerInfo.STATUS_PLAYING
            user.score = 0
            user.freeCount = 0
            self.users.append(user)

        self.stepState.players = self.users
        self.E = self.userMatch.energy
        self.resetFlag = True
        self.stepCount = 0

    def reset_map(self, mapId):  # load map info
        self.map = copy.deepcopy(self.maps[mapId])
        self.userMatch.gameinfo.load_map(self.map)
        self.stepState.golds = self.userMatch.gameinfo.golds
        self.energyOnMap = copy.deepcopy(self.map)
        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if self.map[y][x] > 0:  # gold
                    self.energyOnMap[y][x] = -4
                else:  # obstacles
                    self.energyOnMap[y][x] = ObstacleInfo.types[self.map[y][x]]

    def connect(self):  # simulate player's connect request
        print("Connected to server.")

    def receive(self):  # send data to player (simulate player's receive request)
        if self.resetFlag:  # for the first time -> send game info
            self.resetFlag = False
            data = self.userMatch.to_json()
            return data
        else:  # send step state
            self.stepCount = self.stepCount + 1
            if self.stepCount >= self.maxStep:
                for player in self.stepState.players:
                    player.status = PlayerInfo.STATUS_STOP_END_STEP
            data = self.stepState.to_json()
            return data

    def send(self, message: str):  # receive message from player (simulate send request from player)
        if message.startswith('map'):  # reset game
            requests = message.split(",")
            logging.info("Reset game: %s", message)
            self.reset(requests)
        else:  # player send action
            self.resetFlag = False
            self.stepState.changedObstacles = []
            self.craftUsers = []
            actions_dict = json.loads(message)
            for user in self.users:
                if user.status == PlayerInfo.STATUS_PLAYING:
                    action = actions_dict[str(user.playerId)]
                    user.lastAction = action
                    self.step_action(user, action)
            self.action_5_craft()
            for c in self.stepState.changedObstacles:
                self.map[c["posy"]][c["posx"]] = -c["type"]
                self.energyOnMap[c["posy"]][c["posx"]] = c["value"]

    def step_action(self, user, action):
        switcher = {
            0: self.action_0_left,
            1: self.action_1_right,
            2: self.action_2_up,
            3: self.action_3_down,
            4: self.action_4_free,
            5: self.action_5_craft_pre
        }
        func = switcher.get(action, self.invalidAction)
        func(user)

    def action_5_craft_pre(self, user):  # collect players who craft at current step
        user.freeCount = 0
        if self.map[user.posy][user.posx] <= 0:  # craft at the non-gold cell
            user.energy -= 10
            if user.energy <= 0:
                user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
                user.lastAction = 6 #eliminated
        else:
            user.energy -= 5
            if user.energy > 0:
                self.craftUsers.append(user)
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    count = self.craftMap[key]
                    self.craftMap[key] = count + 1
                else:
                    self.craftMap[key] = 1
            else:
                user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
                user.lastAction = 6 #eliminated

    def action_0_left(self, user):  # user go left
        user.freeCount = 0
        user.posx = user.posx - 1
        if user.posx < 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_1_right(self, user):  # user go right
        user.freeCount = 0
        user.posx = user.posx + 1
        if user.posx >= self.userMatch.gameinfo.width:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_2_up(self, user):  # user go up
        user.freeCount = 0
        user.posy = user.posy - 1
        if user.posy < 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_3_down(self, user):  # user go right
        user.freeCount = 0
        user.posy = user.posy + 1
        if user.posy >= self.userMatch.gameinfo.height:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_4_free(self, user):  # user free
        user.freeCount += 1
        if user.freeCount == 1:
            user.energy += int(self.E / 4)
        elif user.freeCount == 2:
            user.energy += int(self.E / 3)
        elif user.freeCount == 3:
            user.energy += int(self.E / 2)
        else:
            user.energy = self.E
        if user.energy > self.E:
            user.energy = self.E

    def action_5_craft(self):
        craftCount = len(self.craftUsers)
        # print ("craftCount",craftCount)
        if (craftCount > 0):
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                c = self.craftMap[key]
                m = min(math.ceil(self.map[y][x] / c), 50)
                user.score += m
                # print ("user", user.playerId, m)
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    c = self.craftMap[key]
                    del self.craftMap[key]
                    m = min(math.ceil(self.map[y][x] / c), 50)
                    self.map[y][x] -= m * c
                    if self.map[y][x] < 0:
                        self.map[y][x] = 0
                        self.energyOnMap[y][x] = ObstacleInfo.types[0]
                    for g in self.stepState.golds:
                        if g.posx == x and g.posy == y:
                            g.amount = self.map[y][x]
                            if g.amount == 0:
                                self.stepState.golds.remove(g)
                                self.add_changed_obstacle(x, y, 0, ObstacleInfo.types[0])
                                if len(self.stepState.golds) == 0:
                                    for player in self.stepState.players:
                                        player.status = PlayerInfo.STATUS_STOP_EMPTY_GOLD
                            break;
            self.craftMap = {}

    def invalidAction(self, user):
        user.status = PlayerInfo.STATUS_ELIMINATED_INVALID_ACTION
        user.lastAction = 6 #eliminated

    def go_to_pos(self, user):  # player move to cell(x,y)
        if self.map[user.posy][user.posx] == -1:
            user.energy -= randrange(16) + 5
        elif self.map[user.posy][user.posx] == 0:
            user.energy += self.energyOnMap[user.posy][user.posx]
        elif self.map[user.posy][user.posx] == -2:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 0, ObstacleInfo.types[0])
        elif self.map[user.posy][user.posx] == -3:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 3,
                                      self.bog_energy_chain[self.energyOnMap[user.posy][user.posx]])
        else:
            user.energy -= 4
        if user.energy <= 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
            user.lastAction = 6 #eliminated

    def add_changed_obstacle(self, x, y, t, v):
        added = False
        for o in self.stepState.changedObstacles:
            if o["posx"] == x and o["posy"] == y:
                added = True
                break
        if added == False:
            o = {}
            o["posx"] = x
            o["posy"] = y
            o["type"] = t
            o["value"] = v
            self.stepState.changedObstacles.append(o)

    def close(self):
        print("Close socket.")
