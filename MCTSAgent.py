from agent import Agent
import fenix
import time
import math
import random
import numpy as np


        
class MCTSAgent(Agent):
    """
    Agent MCTS avec une rollout policy corresondant à la fonction d'évaluation liée au 
    nombre de pions, généraux et roi
    """
    def __init__(self, player):
        super().__init__(player)
        self.root_node = None
        self.previous_state = None
        self.a = 0

    def act(self, state, remaining_time):
        """
        Chooses the best action for the agent using the Monte Carlo Tree Search algorithm.
        """
        if(self.a<5):
            actions = state.actions()
            if len(actions) == 0:
                raise Exception("No action available.")
            depth = 4
            best_action = None
            best_value = float("-inf")
            alpha = float("-inf")
            beta = float("inf")
            for action in actions:
                value = self.min_value(state.result(action), alpha, beta, depth - 1)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
            self.a += 1
            return best_action
        if self.root_node is None or self.previous_state is None :
            self.root_node = MCTSNode(state, self.player)
        else:

            matched = False
            for child in self.root_node.children:
                if child.state._hash() == state._hash():
                    self.root_node = child
                    matched = True
                    break
            if not matched:
                self.root_node = MCTSNode(state, self.player)
        best_node = self.root_node.best_action()
        self.previous_state = state.result(best_node.parent_action)
        self.root_node = best_node
        return best_node.parent_action
    
    def max_value(self, state, alpha, beta, depth):
        if state.is_terminal() or depth == 0:
            return state.utility(self.player)
        value = float("-inf")
        for action in state.actions():
            eval = self.min_value(state.result(action), alpha, beta, depth - 1)
            value = max(value, eval)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
            
        return value
    
    def min_value(self, state, alpha, beta, depth):
        if state.is_terminal() or depth == 0:
            return self.evaluation(state)
        value = float("inf")
        for action in state.actions():
            eval = self.max_value(state.result(action), alpha, beta, depth - 1)
            value = min(value, eval)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
    
    def evaluation(self, state):
        n_pions,m_pions = list(state.pieces.values()).count(self.player), list(state.pieces.values()).count(-self.player)
        n_general,m_general = list(state.pieces.values()).count(self.player*2), list(state.pieces.values()).count(-self.player*2)
        n_roi, m_roi = list(state.pieces.values()).count(self.player*3), list(state.pieces.values()).count(-self.player*3)
        material_score = n_pions + n_general*2  - m_pions - m_general*2 + n_roi*50 - m_roi*50
        
        return material_score

class MCTSAgent2(Agent):
    """
    Agent MCTS avec une rollout policy aléatoire et qui garde en mémoire l'arbre de recherche
    """
    def __init__(self, player):
        super().__init__(player)
        self.root_node = None
        self.previous_state = None

    def act(self, state, remaining_time):
        """
        Chooses the best action for the agent using the Monte Carlo Tree Search algorithm.
        """
       
        if self.root_node is None or self.previous_state is None :
            self.root_node = MCTSNode2(state, self.player)
        else:

            matched = False
            for child in self.root_node.children:
                if child.state._hash() == state._hash():
                    self.root_node = child
                    matched = True
                    break
            if not matched:
                self.root_node = MCTSNode2(state, self.player)
        best_node = self.root_node.best_action()
        self.previous_state = state.result(best_node.parent_action)
        self.root_node = best_node
        return best_node.parent_action
    
class MCTSAgent3(Agent):
    """ Agent MCAlphaBeta qui garde en mémoire l'arbre de recherche
    """
    def __init__(self, player):
        super().__init__(player)
        self.root_node = None
        self.previous_state = None

    def act(self, state, remaining_time):
        """
        Chooses the best action for the agent using the Monte Carlo Tree Search algorithm.
        """
       
        if self.root_node is None or self.previous_state is None :
            self.root_node = MCTSNode(state, self.player)
        else:

            matched = False
            for child in self.root_node.children:
                if child.state._hash() == state._hash():
                    self.root_node = child
                    matched = True
                    break
            if not matched:
                self.root_node = MCTSNode(state, self.player)
        best_node = self.root_node.best_action()
        self.previous_state = state.result(best_node.parent_action)
        self.root_node = best_node
        return best_node.parent_action   

class MCTSNoMemory(MCTSAgent2):
    """ Agent MCTS sans mémoire
    """
    def __init__(self, player):
        super().__init__(player)
        self.root_node = None
        self.previous_state = None

    def act(self, state, remaining_time):
       
        self.root_node = MCTSNode(state, self.player)
        best_node = self.root_node.best_action()
        return best_node.parent_action

class MCTSNode():
    def __init__(self, state, player,parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.wins = 0
        self.losses = 0
        self.visits = 0
        self.untried_actions = state.actions()
        self.player = player
    
    def q(self):
        return self.wins 
    
    def n(self):
        return self.visits

    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def rollout(self):
        current_state = self.state
        maximizing_player = False
        while not current_state.is_terminal():
            actions = current_state.actions()
            action = self.rollout_policy(actions, current_state, maximizing_player)
            current_state = current_state.result(action)
            maximizing_player = not maximizing_player
        return current_state.utility(self.player) if self.parent else 0       
    
    def backpropagate(self, result):
        self.visits += 1
        if result == 1:
            self.wins += 1
        elif result == -1:
            self.losses += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    def expand(self):
        action = self.untried_actions.pop(len(self.untried_actions) - 1)
        next_state = self.state.result(action)
        child_node = MCTSNode(
            next_state, self.player, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 
    
    def best_child(self, c_param=1.41):
    
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves, state, maximizing_player):
        best_move = possible_moves[0]
        if maximizing_player:
            max = -float('inf')
            for i in range(len(possible_moves)):
                value = self.evaluation(state.result(possible_moves[i]))
                if(value > max):
                    max = value
                    best_move = possible_moves[i]
            return best_move
        else:
            min = float('inf')
            for i in range(len(possible_moves)):
                value = self.evaluation(state.result(possible_moves[i]))
                if(value < min):
                    min = value
                    best_move = possible_moves[i]
            return best_move
    
    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, time_limit=2.0):
        simulation = 0
        start_time = time.perf_counter()
        while True:
            if time.perf_counter() - start_time > time_limit:
                break
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            simulation += 1
        return self.best_child(c_param=0.1)
    
    def is_terminal_node(self):
        return self.state.is_terminal()

    def evaluation(self, state):
        values = list(state.pieces.values())
        n_pions = values.count(self.player)
        m_pions = values.count(-self.player)
        n_general = values.count(self.player * 2)
        m_general = values.count(-self.player * 2)
        n_roi = values.count(self.player * 3)
        m_roi = values.count(-self.player * 3)
        return n_pions + n_general * 2 + n_roi * 50 - (m_pions + m_general * 2 + m_roi * 50) * random.uniform(0.5, 1.5)
    
class MCTSNode2():
    def __init__(self, state, player,parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.wins = 0
        self.losses = 0
        self.visits = 0
        self.untried_actions = state.actions()
        self.player = player
    
    def q(self):
        return self.wins -self.losses
    
    def n(self):
        return self.visits

    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def rollout(self):
        current_state = self.state
        while not current_state.is_terminal():
            actions = current_state.actions()
            action = self.rollout_policy(actions, current_state, )
            current_state = current_state.result(action)
        return current_state.utility(self.player) if self.parent else 0       
    
    def backpropagate(self, result):
        self.visits += 1
        if result == 1:
            self.wins += 1
        elif result == -1:
            self.losses += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    def expand(self):
        action = self.untried_actions.pop(len(self.untried_actions) - 1)
        next_state = self.state.result(action)
        child_node = MCTSNode(
            next_state, self.player, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 
    
    def best_child(self, c_param=1.41):
    
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves, state):
        
        return random.choice(possible_moves)
    
    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, time_limit=2.0):
        simulation = 0
        start_time = time.perf_counter()
        while True:
            if time.perf_counter() - start_time > time_limit:
                break
            v = self._tree_policy()
            reward = v.rollout() 
            v.backpropagate(reward)
            simulation += 1
        return self.best_child(c_param=0.1)
    
    def is_terminal_node(self):
        return self.state.is_terminal()
    

