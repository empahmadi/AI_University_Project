# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import time

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        reward = 0
        # return a very high reward if this action leads to winning.
        if successorGameState.isWin():
            return 99999

        # return a very high penalty for suicide if ghost did not scared.
        for i in range(len(newGhostStates)):
            if newGhostStates[i].getPosition() == newPos and newScaredTimes[i] == 0:
                return -99999

        # a penalty for stopping action.
        if action == 'Stop':
            reward -= 10

        # a penalty for the nearest ghost distance.
        # an appropriate reward for nearest scared ghost.
        for i in range(len(newGhostStates)):
            distance = manhattanDistance(newPos, newGhostStates[i].getPosition())
            if distance > 0:
                if newScaredTimes[i] == 0:
                    reward -= float(1 / distance)
                else:
                    reward += float(2 / distance)

        # a reward for the nearst state to one of the foods.
        # a penalty for number of remaining foods in new states.
        nearestFoodDist = min([util.manhattanDistance(newPos, food) for food in newFood.asList()])
        reward += float(1 / nearestFoodDist)
        reward -= len(newFood.asList())

        # an appropriate reward for nearest capsule.
        # a penalty for number of remaining capsule.
        capsules = successorGameState.getCapsules()
        if len(capsules) > 0:
            nearestCapsuleDist = min([util.manhattanDistance(newPos, capsule) for capsule in capsules])
            reward += float(2 / nearestCapsuleDist)
            reward -= len(capsules)*2

        return reward


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # minimizer node aka one of the ghosts turns.
        def min_value(state, agent_index, depth):
            number_of_agents = gameState.getNumAgents()
            # get the available actions for ghost.
            legal_actions = state.getLegalActions(agent_index)

            # if there is no available actions for ghost
            # return the evaluation function.
            if not legal_actions:
                return self.evaluationFunction(state)

            # if we reach to the last ghost agent then
            # it is the turn of pacman else
            # another ghost should do it's work.
            if agent_index == number_of_agents - 1:
                min_val = min(max_value(state.generateSuccessor(agent_index, ac), depth)
                              for ac in legal_actions)
            else:
                min_val = min(min_value(state.generateSuccessor(agent_index, ac),
                                        agent_index + 1, depth) for ac in legal_actions)

            return min_val

        # maximizer node aka pac-mans turn, hard 0 is the agent index which is the pacman index.
        def max_value(state, depth):
            # get the actions that pacman can do.
            legal_actions = state.getLegalActions(0)

            # if there is no actions left for pacman or
            # we reached at given depth return the evaluation function.
            if not legal_actions or depth == self.depth:
                return self.evaluationFunction(state)

            # finding and getting the maximum value of its actions.
            max_val = max(min_value(state.generateSuccessor(0, ac), 1, depth + 1) for ac in legal_actions)

            return max_val

        # here we are calculating the root of minimax tree.
        # root is pacman and want to choose a path so this node is a maximizer.
        # so the root aka pacman will choose a way with highest score.
        actions = gameState.getLegalActions(0)
        all_actions = {}
        for action in actions:
            all_actions[action] = min_value(gameState.generateSuccessor(0, action), 1, 1)
        return max(all_actions, key=all_actions.get)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # minimizer node aka ghosts turn (alpha beta pruning version)
        def min_value(state, agent_index, depth, alpha, beta):
            number_of_agents = gameState.getNumAgents()
            # get the available actions for ghost.
            legal_actions = state.getLegalActions(agent_index)

            # if there is no available actions for ghost
            # return the evaluation function.
            if not legal_actions:
                return self.evaluationFunction(state)

            # if we reach to the last ghost agent then
            # it is the turn of pacman else
            # another ghost should do it's work.
            # when the alph is greater than the current beta (which is
            # the value of the current node) then it should be pruned.
            min_val = 99999
            current_beta = beta
            if agent_index == number_of_agents - 1:
                for ac in legal_actions:
                    min_val = min(min_val, max_value(state.generateSuccessor(agent_index, ac),
                                                     depth, alpha, current_beta))
                    if min_val < alpha:
                        return min_val
                    current_beta = min(current_beta, min_val)

            else:
                for ac in legal_actions:
                    min_val = min(min_val, min_value(state.generateSuccessor(agent_index, ac),
                                                     agent_index + 1, depth, alpha, current_beta))
                    if min_val < alpha:
                        return min_val
                    current_beta = min(current_beta, min_val)

            return min_val

        # maximizer node aka pacman turns (alpha beta pruning version)
        # hard 0 is the agent index which is the pacman index
        def max_value(state, depth, alpha, beta):
            # get the actions that pacman can do.
            legal_actions = state.getLegalActions(0)

            # if there is no actions left for pacman or
            # we reached at given depth return the evaluation function.
            if not legal_actions or depth == self.depth:
                return self.evaluationFunction(state)

            max_val = -99999
            current_alpha = alpha

            # finding and getting the maximum value of its actions.
            # when the beta is less than the current beta (which is
            # the value of the current node) then it should be pruned.
            for ac in legal_actions:
                max_val = max(max_val, min_value(state.generateSuccessor(0, ac),
                                                 1, depth + 1, current_alpha, beta))
                if max_val > beta:
                    return max_val
                current_alpha = max(current_alpha, max_val)

            return max_val

        # here we are calculating the root of minimax tree.
        # root is pacman and want to choose a path so this node is a maximizer.
        # so the root aka pacman will choose a way with the highest score.
        actions = gameState.getLegalActions(0)

        alpha = -99999
        beta = 99999
        all_actions = {}
        for action in actions:
            value = min_value(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
            all_actions[action] = value

            # update alpha
            if value > beta:
                return action
            alpha = max(value, alpha)

        return max(all_actions, key=all_actions.get)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # under test

        # expect node aka ghosts turn
        def exp_value(state, agent_index, depth):
            number_of_agents = gameState.getNumAgents()
            # get the available actions for ghost.
            legal_actions = state.getLegalActions(agent_index)

            # if there is no available actions for ghost
            # return the evaluation function.
            if not legal_actions:
                return self.evaluationFunction(state)

            # we should define a probability for available actions
            # if we reach to the last ghost agent then
            # it is the turn of pacman else
            # another ghost should do it's work.
            # when the alph is greater than the current beta (which is
            # the value of the current node) then it should be pruned.
            expected_value = 0
            probability = 1.0 / len(legal_actions)

            for ac in legal_actions:
                if agent_index == number_of_agents - 1:
                    currentExpValue = max_value(state.generateSuccessor(agent_index, ac),
                                                depth)
                else:
                    currentExpValue = exp_value(state.generateSuccessor(agent_index, ac),
                                                agent_index + 1, depth)
                expected_value += currentExpValue * probability

            return expected_value

        # maximizer node aka pac-mans turn, hard 0 is the agent index which is the pacman index.
        def max_value(state, depth):
            # get the actions that pacman can do.
            legal_actions = state.getLegalActions(0)

            # if there is no actions left for pacman or
            # we reached at given depth return the evaluation function.
            if not legal_actions or depth == self.depth:
                return self.evaluationFunction(state)

            # finding and getting the maximum value of its actions.
            max_val = max(exp_value(state.generateSuccessor(0, ac), 1, depth + 1) for ac in legal_actions)

            return max_val

        # here we are calculating the root of expectimax tree.
        # root is pacman and want to choose a path so this node is a maximizer.
        # so the root aka pacman will choose a way with the highest score.
        actions = gameState.getLegalActions(0)
        all_actions = {}
        for action in actions:
            all_actions[action] = exp_value(gameState.generateSuccessor(0, action), 1, 1)
        return max(all_actions, key=all_actions.get)

        # end of the test


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    reward = 0
    # return a very high reward for winning state.
    if currentGameState.isWin():
        return 99999

    # return a very high penalty for suicide if ghost did not scared.
    for i in range(len(ghostStates)):
        if ghostStates[i].getPosition() == pacmanPosition and scaredTimers[i] == 0:
            return -99999

    # a penalty for the nearest ghost distance.
    # an appropriate reward for nearest scared ghost.
    for i in range(len(ghostStates)):
        distance = manhattanDistance(pacmanPosition, ghostStates[i].getPosition())
        if distance > 0:
            if scaredTimers[i] == 0:
                reward -= float(1 / distance)
            else:
                reward += float(2 / distance)

    # a reward for the nearst state to one of the foods.
    # a penalty for number of remaining foods in new states.
    nearestFoodDist = min([util.manhattanDistance(pacmanPosition, food) for food in foods.asList()])
    reward += float(1 / nearestFoodDist)
    reward -= len(foods.asList())

    # an appropriate reward for nearest capsule.
    # a penalty for number of remaining capsule.
    capsules = currentGameState.getCapsules()
    if len(capsules) > 0:
        nearestCapsuleDist = min([util.manhattanDistance(pacmanPosition, capsule) for capsule in capsules])
        reward += float(2 / nearestCapsuleDist)
        reward -= len(capsules) * 2

    return reward


# Abbreviation
better = betterEvaluationFunction
