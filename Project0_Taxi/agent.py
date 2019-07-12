import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.08
        self.gamma = 0.8
        self.epsilon = 0.005
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        policy_s = self.epsilon_greedy_probs(self.Q[state])
        return np.random.choice(np.arange(self.nA), p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        Qsa = self.Q[state][action]
        policy_s = self.epsilon_greedy_probs(self.Q[next_state])
        #self.Q[state][action] = Qsa + (self.alpha * (reward + (self.gamma * np.dot(self.Q[next_state], policy_s) ) - Qsa)) # best at around 9.089
        self.Q[state][action] = Qsa + (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state]) ) - Qsa))
        #next_action = self.select_action(next_state)
        #self.Q[state][action] = Qsa + (self.alpha * (reward + (self.gamma * self.Q[next_state][next_action] ) - Qsa))
        
    def epsilon_greedy_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """

        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s