from agent import Agent
import fenix
import time
import math
import random
import numpy as np

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
        return self.wins -self.losses
    
    def n(self):
        return self.visits

    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def rollout(self):
        current_state = self.state
        while not current_state.is_terminal():
            actions = current_state.actions()
            action = self.rollout_policy(actions, current_state)
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
    
    def best_child(self, c_param=0.1):
    
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves, state):
        best_move = possible_moves[0]
        max = -float('inf')
        for i in range(len(possible_moves)):
            if(self.evaluation(state.result(possible_moves[i])) > max):
                max = self.evaluation(state.result(possible_moves[i]))
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
        return n_pions + n_general * 2 + n_roi * 50 - (m_pions + m_general * 2 + m_roi * 50)
        
class MCTSAgent(Agent):

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

class MCTSNode3(MCTSNode):
    def rollout(self):
        return self.alphabeta(self.state, depth=3, alpha=-float('inf'), beta=float('inf'))

    def alphabeta(self, state, depth, alpha, beta):
        if depth == 0 or state.is_terminal():
            return state.utility(self.player)
        for action in state.actions():
            next_state = state.result(action)
            value = -self.alphabeta(next_state, depth-1, -beta, -alpha)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return alpha

class MCTSAgent3(MCTSAgent):

    def __init__(self, player):
        super().__init__(player)
        self.root_node = None
        self.previous_state = None

    def act(self, state, remaining_time):
        """
        Chooses the best action for the agent using the Monte Carlo Tree Search algorithm.
        """
       
        if self.root_node is None or self.previous_state is None :
            self.root_node = MCTSNode3(state, self.player)
        else:

            matched = False
            for child in self.root_node.children:
                if child.state._hash() == state._hash():
                    self.root_node = child
                    matched = True
                    break
            if not matched:
                self.root_node = MCTSNode3(state, self.player)
        best_node = self.root_node.best_action()
        self.previous_state = state.result(best_node.parent_action)
        self.root_node = best_node
        return best_node.parent_action

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
            action = self.rollout_policy(actions, current_state)
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
    
    def best_child(self, c_param=0.1):
    
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


class MCTSAgent2(Agent):

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
    


class MCTSNode4(MCTSNode):
    def rollout(self):
        return self.alphabeta(self.state, depth=3, alpha=-float('inf'), beta=float('inf'))

    def alphabeta(self, state, depth, alpha, beta):
        if depth == 0 or state.is_terminal():
            return state.utility(self.player)
        for action in state.actions():
            next_state = state.result(action)
            value = -self.alphabeta(next_state, depth-1, -beta, -alpha)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return alpha

class MCTSAgent4(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.root_node = None
        self.previous_state = None

    def act(self, state, remaining_time):
        """
        Chooses the best action for the agent using the Monte Carlo Tree Search algorithm.
        """
       
        if self.root_node is None or self.previous_state is None :
            self.root_node = MCTSNode4(state, self.player)
        else:

            matched = False
            for child in self.root_node.children:
                if child.state._hash() == state._hash():
                    self.root_node = child
                    matched = True
                    break
            if not matched:
                self.root_node = MCTSNode4(state, self.player)
        best_node = self.root_node.best_action()
        self.previous_state = state.result(best_node.parent_action)
        self.root_node = best_node
        return best_node.parent_action