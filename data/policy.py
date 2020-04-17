import numpy as np


class SwingUpAndBalancePolicy(object):
    def __init__(self, weights_file):
        d = np.load(weights_file)
        self.fc1_w = d['fc1_w']
        self.fc1_b = d['fc1_b']
        self.fc2_w = d['fc2_w']
        self.fc2_b = d['fc2_b']
        self.action_w = d['action_w']
        self.action_b = d['action_b']
        self.mean = d['mean']
        self.stddev = d['stddev']

    def normalize_state(self, state):
        # Convert the state representation to the one used by the Gym Env CartpoleSwingUp
        theta_dot, x_dot, theta, x_pos = state
        theta += np.pi
        result = (np.array([[x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]]) - self.mean) / self.stddev
        return np.clip(result[0], -5, 5)

    def predict(self, state):
        state = self.normalize_state(state)
        x = np.tanh(self.fc1_w @ state + self.fc1_b)
        x = np.tanh(self.fc2_w @ x + self.fc2_b)
        x = self.action_w @ x + self.action_b
        return x[0]


class RandomPolicy(object):
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def predict(self, state):
        return self.rng.uniform(-1.0, 1.0)
