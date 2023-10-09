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


from util import manhattanDistance
from game import Directions
import random, util
import math
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        distances = []

        # find distances for all foods
        for food in currentGameState.getFood().asList():
            distances.append(-(manhattanDistance(newPos, food)))

        # check if any ghosts are near
        for state in newGhostStates:
            if state.getPosition() == newPos and state.scaredTimer is 0:
                return -math.inf

        return max(distances)

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
        self.index = 0 # Pacman is always agent index 0
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

        # return minimax action
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, currentIndex, currentDepth):
        # return minimax action as [value, action]

        # if successor is pacman, update index & depth
        if currentIndex >= gameState.getNumAgents():
            currentIndex = 0
            currentDepth += 1

        if not gameState.getLegalActions(currentIndex) or currentDepth == self.depth:
            return self.evaluationFunction(gameState), ""
        # get max value for pacman
        if currentIndex == 0:
            return self.maxValue(gameState, currentIndex, currentDepth)

        # get min value for ghosts
        else:
            return self.minValue(gameState, currentIndex, currentDepth)

    def minValue(self, gameState, currentIndex, currentDepth):
        minValue = math.inf
        minMove = ""

        # check successors for each legal move
        for move in gameState.getLegalActions(currentIndex):
            successorDepth = currentDepth
            successorIndex = currentIndex + 1

            currentValue = self.value(gameState.generateSuccessor(currentIndex, move), successorIndex, successorDepth)[0]

            if currentValue < minValue:
                minValue = currentValue
                minMove = move

        return [minValue, minMove]

    def maxValue(self, gameState, currentIndex, currentDepth):
        maxValue = -math.inf
        maxMove = ""

        # check successors for each legal move
        for move in gameState.getLegalActions(currentIndex):
            successorDepth = currentDepth
            successorIndex = currentIndex + 1

            currentValue = self.value(gameState.generateSuccessor(currentIndex, move), successorIndex, successorDepth)[0]

            if currentValue > maxValue:
                maxValue = currentValue
                maxMove = move

        return [maxValue, maxMove]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0, -math.inf, math.inf)[1]

    def value(self, gameState, currentIndex, currentDepth, alpha, beta):
        # return minimax action as [value, action]

        # if successor is pacman, update index & depth
        if currentIndex >= gameState.getNumAgents():
            currentIndex = 0
            currentDepth += 1

        if not gameState.getLegalActions(currentIndex) or currentDepth == self.depth:
            return self.evaluationFunction(gameState), ""

        # get max value for pacman
        if currentIndex == 0:
            return self.maxValue(gameState, currentIndex, currentDepth, alpha, beta)

        # get min value for ghosts
        else:
            return self.minValue(gameState, currentIndex, currentDepth, alpha, beta)

    def minValue(self, gameState, currentIndex, currentDepth, alpha, beta):
        minValue = math.inf
        minMove = ""

        # check successors for each legal move
        for move in gameState.getLegalActions(currentIndex):
            successorDepth = currentDepth
            successorIndex = currentIndex + 1

            currentValue = self.value(gameState.generateSuccessor(currentIndex, move), successorIndex,
                                      successorDepth, alpha, beta)[0]

            if currentValue < minValue:
                minValue = currentValue
                minMove = move

            # prune because minValue is less than alpha and we are trying to maximize
            if minValue < alpha:
                return [minValue, minMove]

            # update beta
            beta = min(beta, minValue)
        return [minValue, minMove]

    def maxValue(self, gameState, currentIndex, currentDepth, alpha, beta):
        maxValue = -math.inf
        maxMove = ""

        # check successors for each legal move
        for move in gameState.getLegalActions(currentIndex):
            successorDepth = currentDepth
            successorIndex = currentIndex + 1

            currentValue = self.value(gameState.generateSuccessor(currentIndex, move), successorIndex,
                                      successorDepth, alpha, beta)[0]

            if currentValue > maxValue:
                maxValue = currentValue
                maxMove = move

            # prune because max value is greater than beta and we are trying to minimize
            if maxValue > beta:
                return [maxValue, maxMove]

            # update alpha
            alpha = max(alpha, maxValue)

        return [maxValue, maxMove]


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
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, currentIndex, currentDepth):
        # return minimax action as [value, action]

        # if successor is pacman, update index & depth
        if currentIndex >= gameState.getNumAgents():
            currentIndex = 0
            currentDepth += 1

        if not gameState.getLegalActions(currentIndex) or currentDepth == self.depth:
            return self.evaluationFunction(gameState), ""
        # get max value for pacman
        if currentIndex == 0:
            return self.maxValue(gameState, currentIndex, currentDepth)

        # get exp value for ghosts
        else:
            return self.expValue(gameState, currentIndex, currentDepth)

    def maxValue(self, gameState, currentIndex, currentDepth):
        maxValue = -math.inf
        maxMove = ""

        # check successors for each legal move
        for move in gameState.getLegalActions(currentIndex):
            successorDepth = currentDepth
            successorIndex = currentIndex + 1

            currentValue = self.value(gameState.generateSuccessor(currentIndex, move), successorIndex, successorDepth)[0]

            if currentValue > maxValue:
                maxValue = currentValue
                maxMove = move

        return [maxValue, maxMove]

    def expValue(self, gameState, currentIndex, currentDepth):
        expValue = 0
        expMove = ""

        # explore all successor moves
        for move in gameState.getLegalActions(currentIndex):
            successorDepth = currentDepth
            successorIndex = currentIndex + 1

            currentValue = self.value(gameState.generateSuccessor(currentIndex, move), successorIndex, successorDepth)[0]

            # get expected value for each move by E(X) = sum((probability_x * value_x))
            # in this case, the probabilities are uniformly distributed, so all actions are equally probable to occur
            expValue += (1.0 / len(gameState.getLegalActions(currentIndex))) * currentValue

        return [expValue, expMove]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
                    this evaluation function analyzes each variable in gameState to determine which steps to take.
                    it evaluates based on: current position, closest foods, current uneaten pellets (50 pts each),
                                        closest ghost, current score, and number of scared ghosts


    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFoods = currentGameState.getFood().asList()
    currentPellets = currentGameState.getCapsules()
    currentGhosts = currentGameState.getGhostStates()
    currentScore = currentGameState.getScore()

    # calculate distances to each food
    foodDistances = [-manhattanDistance(currentPos, position) for position in currentFoods]
    foodDistances = foodDistances if len(foodDistances) else [0]

    # calculate distances to each ghost
    scaredGhosts = [ghost for ghost in currentGhosts if ghost.scaredTimer is not 0]
    ghostDistances = [0 for ghost in scaredGhosts]
    for ghost in currentGhosts:
        distance = manhattanDistance(currentPos, ghost.getPosition())
        ghostDistances.append(0 if distance == 0 else -distance)

    # return closest food, closest ghost, score, number of pellets left, and number of scared ghosts
    return max(foodDistances) + min(ghostDistances) + currentScore - 50*(len(currentPellets)) - (len(scaredGhosts))


# Abbreviation
better = betterEvaluationFunction



