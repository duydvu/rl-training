from enum import Enum

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    REST = 4
    MINE = 5


class Obstacle(Enum):
    Tree = 1
    Trap = 2
    Swamp = 3
