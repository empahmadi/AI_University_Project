from graphicsUtils import sleep
from pacman import Directions
from game import Actions
from pacman import GameState
from graphicsDisplay import PacmanGraphics

game_state = GameState()

# directions of moves
closestDotEatingPath = [Directions.EAST, Directions.EAST, Directions.NORTH, Directions.WEST, Directions.WEST,
                        Directions.NORTH, Directions.NORTH, Directions.EAST, Directions.EAST, Directions.SOUTH,
                        Directions.SOUTH, Directions.EAST, Directions.EAST, Directions.NORTH, Directions.WEST,
                        Directions.WEST, Directions.SOUTH, Directions.SOUTH, Directions.SOUTH, Directions.EAST,
                        Directions.EAST, Directions.NORTH, Directions.NORTH, Directions.WEST, Directions.WEST,
                        Directions.WEST, Directions.NORTH, Directions.NORTH, Directions.EAST, Directions.EAST,
                        Directions.EAST, Directions.NORTH, Directions.WEST, Directions.WEST, Directions.SOUTH,
                        Directions.EAST, Directions.NORTH]

shortestPath = [Directions.EAST, Directions.EAST, Directions.EAST, Directions.EAST, Directions.NORTH, Directions.NORTH,
                Directions.NORTH, Directions.NORTH, Directions.WEST, Directions.WEST, Directions.WEST, Directions.WEST,
                Directions.WEST, Directions.WEST, Directions.WEST, Directions.SOUTH, Directions.SOUTH, Directions.SOUTH,
                Directions.SOUTH, Directions.EAST, Directions.EAST, Directions.EAST, Directions.EAST, Directions.EAST,
                Directions.EAST, Directions.NORTH, Directions.NORTH, Directions.WEST, Directions.WEST, Directions.WEST,
                Directions.WEST, Directions.SOUTH, Directions.SOUTH, Directions.SOUTH, Directions.EAST, Directions.EAST,
                Directions.EAST, Directions.EAST, Directions.EAST, Directions.EAST, Directions.NORTH, Directions.NORTH,
                Directions.WEST, Directions.WEST, Directions.WEST, Directions.WEST, Directions.WEST, Directions.SOUTH,
                Directions.SOUTH, Directions.SOUTH, Directions.SOUTH, Directions.EAST, Directions.EAST, Directions.EAST,
                Directions.EAST, Directions.EAST, Directions.NORTH, Directions.NORTH, Directions.WEST, Directions.WEST,
                Directions.WEST, Directions.WEST, Directions.SOUTH, Directions.SOUTH, Directions.EAST, Directions.EAST,
                Directions.NORTH]


# execute actions in the game
def executeActions(path):
    for action in path:
        legal = game_state.getLegalPacmanActions()
        if action not in legal:
            action = Directions.STOP
        state = game_state.generatePacmanSuccessor(action)
        sleep(0.1)
        graphics = PacmanGraphics(1.0)
        graphics.initialize(game_state.data)
        graphics.startGraphics(state)
        print("reached")
