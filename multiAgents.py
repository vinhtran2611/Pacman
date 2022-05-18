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


from nbformat import current_nbformat
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED
from util import manhattanDistance
from game import Directions
import random, util
from pacman import GameState
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
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        diff_pacman_ghosts = [manhattanDistance(newPos, ghostState.configuration.pos) for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        min_gh_pc_diff = 999999
        if(len(diff_pacman_ghosts) > 0):
            min_gh_pc_diff = min(diff_pacman_ghosts)
        min_food_dist = None
        manhattan_foods = [manhattanDistance(newPos, pos) for pos in currentFood.asList() if currentFood[pos[0]][pos[1]]]
        min_food_dist = min(manhattan_foods)
        return - 150 / (1 + min_gh_pc_diff)  + 50 / (1 + min_food_dist)
        return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def Minimax(self, gameState, current_depth):
        print(f'\nSelf.depth = {self.depth}, current_depth = {current_depth} -------------------------------------')
        total_agents = gameState.getNumAgents()
        agentIdx = current_depth % total_agents - 1
        
        if (agentIdx == 0):
            # pacman!
            print(f'I am Pacman!!!!!')
            moves = gameState.getLegalActions(agentIdx)
            print(f'Pacman at {current_depth} can move: {moves}')
            values = []
            for move in moves:
                print(f'\nPacman at {current_depth} Move: {move}')
                successorState = gameState.generateSuccessor(agentIdx, move)
                if (successorState.isLose() or successorState.isWin() or current_depth == self.depth * total_agents):
                    print(f'Ghost don\'t move {current_depth} == {self.depth} * {total_agents}!!')
                    values.append((move, self.evaluationFunction(successorState)))
                else:
                    print(f'Pacman call self.Minimax(successorState, {current_depth} + 1)')
                    next = self.Minimax(successorState, current_depth + 1)[1]
                    values.append((move, next))
            print(f'Pacman Values at {current_depth}: {values}')
            goldenMove = max(values, key=lambda pair: pair[1])
            print(f'Values of Pacman at {current_depth} choosed: {goldenMove}')
            if current_depth == 1:
                print("=================================================DONE==================================================")
            return goldenMove
        else:
            print(f'I am Ghost!!!!!!!!!!!')
            moves = gameState.getLegalActions(agentIdx)
            print(f'Ghost at {current_depth} can move: {moves}')
            values = []
            for move in moves:
                print(f'\nGhost at {current_depth} Move: {move}')
                successorState = gameState.generateSuccessor(agentIdx, move)
                if (successorState.isLose() or successorState.isWin() or current_depth == self.depth * total_agents):
                    print(f'Pacman don\'t move {current_depth} == {self.depth} * {total_agents}!!')
                    values.append((move, self.evaluationFunction(successorState)))
                else:
                    print(f'Ghost call self.Minimax(successorState, {current_depth} + 1)')
                    next = self.Minimax(successorState, current_depth + 1)[1]
                    values.append((move, next))
            print(f'Ghost Values at {current_depth}: {values}')
            goldenMove = min(values, key=lambda pair: pair[1])
            print(f'Values of Ghost at {current_depth} choosed: {goldenMove}')
            return goldenMove

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
        return self.Minimax(gameState, 1)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def Minimax(self, gameState, depth, alpha, beta):
        total_agents = gameState.getNumAgents()
        agentIdx = depth % total_agents - 1
        if(agentIdx == 0):
            #maximizer
            values = []
            actions = gameState.getLegalActions(agentIdx)
            for action in actions:
                successorState = gameState.generateSuccessor(agentIdx, action)
                if(successorState.isLose() or successorState.isWin() or depth == self.depth * total_agents):
                    v = self.evaluationFunction(successorState)
                else:
                    v = self.Minimax(successorState, depth + 1, alpha, beta)[1]
                values.append((action, v))
                if(v > beta):
                    break
                alpha = max(alpha, v)
            goldenMove = max(values, key=lambda x: x[1])
            return goldenMove
        else:
            #minimizer
            values = []
            actions = gameState.getLegalActions(agentIdx)
            for action in actions:
                successorState = gameState.generateSuccessor(agentIdx, action)
                if (successorState.isLose() or successorState.isWin() or depth == self.depth * total_agents):
                    v = self.evaluationFunction(successorState)
                else:
                    v = self.Minimax(successorState, depth + 1, alpha, beta)[1]
                values.append((action, v))
                if (v < alpha):
                    break
                beta = min(beta, v)
            goldenMove = min(values, key=lambda x: x[1])
            return goldenMove

    def getAction(self, gameState):
        return self.Minimax(gameState, 1, -99999999, 99999999)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def Expectimax(self, gameState, current_depth):
        total_agents = gameState.getNumAgents()
        agentIdx = current_depth % total_agents - 1
        if (agentIdx == 0):
            # pacman!
            moves = gameState.getLegalActions(agentIdx)
            values = []
            for move in moves:
                successorState = gameState.generateSuccessor(agentIdx, move)
                if (successorState.isLose() or successorState.isWin() or current_depth == self.depth * total_agents):
                    values.append((move, self.evaluationFunction(successorState)))
                else:
                    values.append((move ,self.Expectimax(successorState, current_depth + 1)[1]))
            goldenMove = max(values, key=lambda pair: pair[1])
            return goldenMove
        else:
            actions = gameState.getLegalActions(agentIdx)
            sum = 0
            for action in actions:
                successorState = gameState.generateSuccessor(agentIdx, action)
                if (successorState.isLose() or successorState.isWin() or current_depth == self.depth * total_agents):
                    sum += self.evaluationFunction(successorState)
                else:
                    sum += self.Expectimax(successorState, current_depth + 1)[1]
            goldenMove = (None, sum / len(actions))
            return goldenMove

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(gameState, 1)[0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    capsules = currentGameState.getCapsules()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostsPos = [ghost.configuration.pos for ghost in ghostStates]
    pacmanPos = currentGameState.getPacmanPosition()
    total_score = 0
    distances = mazeDistances(pacmanPos, currentGameState)
    #get minimum distance to food
    food_infulence = 100
    pacman_distance_to_min_food = None
    total_food_distances = 0

    for x, y in foodGrid.asList():
        if(foodGrid[x][y] ):
            maze_distance = distances[(x, y)]
            total_food_distances += maze_distance
            if(pacman_distance_to_min_food is None or pacman_distance_to_min_food < maze_distance):
                pacman_distance_to_min_food = maze_distance
    total_score = total_score + food_infulence / (1 + total_food_distances)
    total_score += food_infulence / (1 + currentGameState.getNumFood())
    #not afraid ghosts distance
    not_afraid_ghost_influence = -6
    afraid_ghost_influence = 200
    total_not_afraid_ghosts_distance = 0
    min_afraid_ghosts_distance = None
    for idx, ghostPos in enumerate(ghostsPos):
        maze_distance = manhattanDistance(pacmanPos, ghostPos)
        if(ghostStates[idx].scaredTimer <= 0):
            total_not_afraid_ghosts_distance += maze_distance
        elif(min_afraid_ghosts_distance is None or min_afraid_ghosts_distance > maze_distance):
            min_afraid_ghosts_distance = maze_distance
    avg_not_afraid_ghosts_dist = total_not_afraid_ghosts_distance / len(ghostsPos)
    total_score = total_score + not_afraid_ghost_influence / (1 + avg_not_afraid_ghosts_dist)
    if(min_afraid_ghosts_distance is not None):
        total_score = total_score + afraid_ghost_influence / (1 + min_afraid_ghosts_distance)
    #Capsules distances
    minimum_v = None
    capsules_distance_influence = 150
    for pellet in capsules:
        total_ghost_distances = 0
        for ghostPos in ghostsPos:
            total_ghost_distances += manhattanDistance(ghostPos, pellet)
        avg_ghost_dist = total_ghost_distances / len(ghostsPos)
        pacman_distance = distances[pellet]
        v = pacman_distance
        if(minimum_v == None or minimum_v > v):
            minimum_v = None
    if (minimum_v is not None):
        total_score = total_score + (capsules_distance_influence)/ (1 + minimum_v)

    return total_score

    util.raiseNotDefined()

def mazeDistances(pacmanPosition, gameState):
    walls = gameState.getWalls()
    queue = util.Queue()
    queue.push((pacmanPosition, 0))
    distances = {}
    distances[pacmanPosition] = 0
    while not(queue.isEmpty()):
        current_node, depth = queue.pop()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            x,y = current_node
            x_new, y_new = (x + dx, y + dy)
            if (not(walls[x_new][y_new]) and (x_new, y_new) not in distances):
                distances[(x_new, y_new)] = depth + 1
                queue.push(((x_new, y_new), depth + 1))
    return distances

# Abbreviation
better = betterEvaluationFunction
