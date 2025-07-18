import numpy as np



class Memory():

    def __init__(self, processes : int, max_memory : int, state_shape : tuple, n_actions : int) -> None:
        self.processes = processes
        self.max_memory = max_memory
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.states = np.zeros((self.max_memory, *state_shape), dtype=np.float32)
        self.actions = np.zeros((self.max_memory, n_actions), dtype=np.int8)
        self.rewards = np.zeros(self.max_memory, dtype=np.float32)
        self.new_states = np.zeros((self.max_memory, *state_shape), dtype=np.float32)
        self.terminates = np.zeros(self.max_memory, dtype=np.int8)
        self.index = 0
        self.counter = 0
        self.count = 0


    def store(self, state : np.ndarray, action : int, reward : float, new_state : np.ndarray, terminate : bool) -> None:
        for i in range(self.processes):
            self.index = self.counter % self.max_memory
            self.states[self.index] = state[i].copy()
            a = np.zeros(self.n_actions, dtype=np.int8)
            a[action[i]] = 1
            self.actions[self.index] = a
            self.rewards[self.index] = reward[i]
            self.new_states[self.index] = new_state[i].copy()
            self.terminates[self.index] = 0 if terminate[i] else 1
            self.counter += 1
            self.count = min(self.counter, self.max_memory)


    def sample(self, n_samples : int, offset : int):
        idx = np.random.choice(self.count, n_samples - offset, replace=False)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        new_states = self.new_states[idx]
        terminates = self.terminates[idx]

        for i in range(offset):
            states = np.vstack((states, self.states[self.index-i]))
            actions = np.vstack((actions, self.actions[self.index-i]))
            rewards = np.append(rewards, self.rewards[self.index-i])
            new_states = np.vstack((new_states, self.new_states[self.index-i]))
            terminates = np.append(terminates, self.terminates[self.index-i])

        return states, actions, rewards, new_states, terminates
