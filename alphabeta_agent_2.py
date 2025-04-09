from agent import Agent
import fenix
import math

class AlphaBetaAgent_2(Agent):

    def act(self, state, remaining_time):

        ### en jouant avec les poids, on tombe sur la même conclusion car la fonction d'évaluation privilégiera les reconstructions !!
        ### je perds du temps bêtement
        """
        ### PRENDRE EN COMPTE REMAINING TIME !!!
        Sélectionne les actions avec comme ordre de priorité capture > reconstruction roi >
        reconstruction général > déplacement simple

        Args :
            state : l'état actuel FenixState
            remaining_time : Le temps restant au moment de jouer

        Returns :
            Une liste des actions parmi lesquelles choisir

        """

        actions = state.actions()
        if not actions:
            raise Exception("No action available")

        # Ordre de priorité sur les actions

        capture_actions = [a for a in actions if a.removed]
        if capture_actions:
            return self.alphaBeta(capture_actions, state)

        king_actions = [a for a in actions if self.isKingReconstruction(state, a)]
        if king_actions:
            return self.alphaBeta(king_actions, state)

        general_actions = [a for a in actions if self.isGeneralReconstruction(state, a)]
        if general_actions:
            return self.alphaBeta(general_actions, state)

        return self.alphaBeta(actions, state)

    def alphaBeta(self, actions, state):
        """
        Applique alpha-Beta avec une limite de profondeur de 3 sur l'ensemble des actions

        Args :
            actions : une liste de FenixAction
            state : l'état actuel FenixState

        Returns :
            La meilleure action
        """
        best_action, best_value = None, float("-inf")
        alpha, beta = float("-inf"), float("inf")
        depth = 3
        for action in actions:
            value = self.min_value(state.result(action), alpha, beta, depth-1)
            if value > best_value:
                best_value, best_action = value, action
        return best_action

    def isGeneralReconstruction(self, state, action):
        """
        Vérifie si l'action crée un général (soldat + soldat) à partir de l'état actuel

        Args :
            state : l'état actuel FenixState
            action : l'action à analyser
        Return :
            True si la création d'un général est possible, False sinon
        """
        if action.removed:
            return False
        start_val = state.pieces.get(action.start, 0)
        end_val = state.pieces.get(action.end, 0)
        return (state.can_create_general
                and abs(start_val) == 1
                and abs(end_val) == 1
                and start_val * state.current_player > 0
                and end_val * state.current_player > 0)

    def isKingReconstruction(self, state, action):
        """
        Vérifie si l'action crée un roi (soldat + soldat) à partir de l'état actuel

        Args :
            state : l'état actuel FenixState
            action : l'action à analyser
        Return :
            True si la création d'un roi est possible, False sinon
        """
        if action.removed:
            return False
        start_val = state.pieces.get(action.start, 0)
        end_val = state.pieces.get(action.end, 0)
        return (state.can_create_king
                and abs(start_val) == 1
                and abs(end_val) == 2
                and start_val * state.current_player > 0
                and end_val * state.current_player > 0)

    def max_value(self, state, alpha, beta, depth):
        """
        Args :
            state : l'état actuel FenixState
            alpha : borne inférieure
            beta : borne supérieure
            depth : profondeur limite de l'algorithme

        Returns :
            La valeur de l'action qui maximise les chances de gagner
        """

        if self.isCutoff(state, depth):
            return self.evalTanh10(state)
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
        Args :
            state : l'état actuel FenixState
            alpha : borne inférieure
            beta : borne supérieure
            depth : profondeur limite de l'algorithme

        Returns :
            La valeur de l'action qui maximise les chances de gagner
        """

        if self.isCutoff(state, depth):
            return self.evalTanh10(state)
        value = float("inf")
        for action in state.actions():
            eval = self.max_value(state.result(action), alpha, beta, depth - 1)
            value = min(value, eval)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def isCutoff(self, state, depth):
        if depth == 0 or state.is_terminal():
            return True
        return False


    def evalNorm(self, state):
        """
        Fonction d'évaluation avec normalisation linéaire sur la valeur matérielle des pièces,
        avec des poids différents entre pions, généraux et roi

        Args :
            state : état actuel FenixState

        Returns :
            Renvoie la fonction d'utilité si l'état est terminal (-1 en cas de défaite, 1 en cas de victoire)
            Renvoie une valeur située entre [-1, 1]

        """

        if state.is_terminal():
            return state.utility(self.player)


        n_pions, m_pions = list(state.pieces.values()).count(-self.player), list(state.pieces.values()).count(1)
        n_general, m_general = list(state.pieces.values()).count(-self.player * 2), list(state.pieces.values()).count(2)
        n_roi, m_roi = list(state.pieces.values()).count(-self.player * 3), list(state.pieces.values()).count(3)

        score_brut = (n_pions - m_pions) + (n_general * 3 - m_general * 3) + (n_roi * 50 - m_roi * 50)
        score_normalise = max(-1.0, min(1.0, score_brut/15))
        return score_normalise


    def evalTanh10(self, state):
        """
        Fonction d'évaluation avec normalisation linéaire sur la valeur matérielle des pièces,
        avec des poids différents entre pions, généraux et roi

        Args :
            state : état actuel FenixState

        Returns :
            Renvoie la fonction d'utilité si l'état est terminal (-1 en cas de défaite, 1 en cas de victoire)
            Renvoie une valeur située entre [-1, 1]

        """

        if state.is_terminal():
            return state.utility(self.player)

        n_pions, m_pions = list(state.pieces.values()).count(-self.player), list(state.pieces.values()).count(1)
        n_general, m_general = list(state.pieces.values()).count(-self.player * 2), list(state.pieces.values()).count(2)
        n_roi, m_roi = list(state.pieces.values()).count(-self.player * 3), list(state.pieces.values()).count(3)
        score_brut = (n_pions - m_pions) + (n_general * 2 - m_general * 2) + (n_roi * 50 - m_roi * 50)
        score_normalise = math.tanh(score_brut / 10)
        return score_normalise

    def evalTanh5(self, state):
        """
        Fonction d'évaluation avec normalisation linéaire sur la valeur matérielle des pièces,
        avec des poids différents entre pions, généraux et roi

        Args :
            state : état actuel FenixState

        Returns :
            Renvoie la fonction d'utilité si l'état est terminal (-1 en cas de défaite, 1 en cas de victoire)
            Renvoie une valeur située entre [-1, 1]

        """

        if state.is_terminal():
            return state.utility(self.player)

        n_pions, m_pions = list(state.pieces.values()).count(-self.player), list(state.pieces.values()).count(1)
        n_general, m_general = list(state.pieces.values()).count(-self.player * 2), list(state.pieces.values()).count(2)
        n_roi, m_roi = list(state.pieces.values()).count(-self.player * 3), list(state.pieces.values()).count(3)
        score_brut = (n_pions - m_pions) + (n_general * 2 - m_general * 2) + (n_roi * 50 - m_roi * 50)
        score_normalise = math.tanh(score_brut / 5)
        return score_normalise


