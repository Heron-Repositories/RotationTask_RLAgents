
import numpy as np
import random
from numpy_default_dict import NumpyDefaultDict


class QLearner:
    def __init__(self, alpha=0.1, gamma=0.9, starting_epsilon=0.9, actions_set=[],
                 epsilon_decay=1e-4, minimum_epsilon=0.1, q_table=None):

        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = starting_epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.actions_set = actions_set

        if q_table is None:
            self.create_q_table()
        else:
            self.q_table = q_table

    def create_q_table(self):
        action_size = len(self.actions_set)
        self.q_table = NumpyDefaultDict(len_of_numpy_array=action_size)

    def learn(self, obs, action, reward, next_obs):
        q_sa_next = np.max(self.q_table[obs]) if next_obs is not None else 0
        td_target = reward + self.gamma * np.max(q_sa_next)
        action_index = np.argwhere(np.in1d(self.actions_set, action))[0][0]
        td_error = td_target - self.q_table[obs][action_index]
        self.q_table[obs][action_index] = self.q_table[obs][action_index] + self.alpha * td_error

    def act(self, observation_state_index):
        if self.epsilon > self.minimum_epsilon:
            self.epsilon -= self.epsilon_decay * self.epsilon
        if np.random.random() > self.epsilon:
            action = self.actions_set[np.argmax(self.q_table[observation_state_index])]
        else:  # Choose a random action
            action = random.choice(self.actions_set)

        return action



