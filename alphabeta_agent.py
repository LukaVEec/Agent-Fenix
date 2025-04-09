from agent import Agent
import fenix
import time
import math
import random
import numpy as np

class AlphaBetaAgent(Agent):
    
    def act(self, state, remaining_time):
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

        return best_action
    
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
    
