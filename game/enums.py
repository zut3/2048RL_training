from enum import IntEnum, Enum


class Direction(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3

class TurnMove(Enum):
    Player = 0
    Game = 1