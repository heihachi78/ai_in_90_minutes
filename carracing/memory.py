import numpy as np



class Memory():

    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.counter = 0
        self.index = 0
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.state_memory = np.zeros((self.mem_size, 3, input_shape[0]), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 3, input_shape[0]), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)
        self.terminate_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.episode_memory = np.ones(self.mem_size, dtype=np.int16) * -1
        self.episode_index = 0
        self.valid_training_data = 0


    def store(self, state, action, new_state, terminated, episode):
        index = self.counter % self.mem_size

        self.episode_memory[index] = episode

        self.state_memory[index][0] = state.copy()
        if self.episode_index > 0:
            self.state_memory[index][1] = self.state_memory[index-1][0].copy()
        if self.episode_index > 1:
            self.state_memory[index][2] = self.state_memory[index-2][0].copy()

        self.new_state_memory[index][0] = new_state.copy()
        if self.episode_index > 0:
            self.new_state_memory[index][1] = state.copy()
        if self.episode_index > 1:
            self.new_state_memory[index][2] = self.state_memory[index-1][0].copy()

        a = np.zeros(self.action_memory.shape[1])
        a[action] = 1.0
        self.action_memory[index] = a

        self.terminate_memory[index] = 0. if terminated else 1.

        self.counter += 1
        self.episode_index += 1
        self.index = index


    def sample_buffer(self, batch_size):
        idx = np.argwhere(self.episode_memory > -1)
        batch = np.random.choice(idx.reshape(idx.shape[0]), min(idx.shape[0], batch_size))
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        states_ = self.new_state_memory[batch]
        terminate = self.terminate_memory[batch]
        return states, actions, states_, terminate, batch


    def get_prediction_state(self):
        return np.expand_dims(self.new_state_memory[self.index], axis=0)

    
    def delete_episode_data(self, episode):
        batch = self.episode_memory[self.episode_memory == episode]
        self.episode_memory[self.episode_memory == episode] = -1
        return batch.shape[0]


    def get_valid_training_data_count(self):
        return np.argwhere(self.episode_memory > -1).shape[0]
