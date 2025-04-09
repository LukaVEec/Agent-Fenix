from agent import Agent
import random
import fenix


class PruningAgent(Agent):
    """
    Class that implements an agent using the minimax algorithm with alpha-beta pruning.
    """
    def act(self, state, remaining_time):
        """
        Chooses the best action for the agent using the minimax algorithm with alpha-beta pruning.
        """
        actions = state.actions()
        if len(actions) == 0:
            raise Exception("No action available.")
        depth = 3
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
        """
        Maximizes the value of the state for the agent.
        """
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
        """
        Minimizes the value of the state for the opponent.
        """
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