import numpy as np



class Driver():

    def __init__(self, environment, launch_control, n_actions=5, data_collection_episodes=0):
        self.env = environment
        self.launch_control = launch_control
        self.n_actions = n_actions
        self.action_stat = np.zeros(n_actions, dtype=np.int32)
        self.action_stat_percentage = np.zeros(n_actions, dtype=np.int32)
        self.data_collection_episodes = data_collection_episodes


    def _get__random_action(self):
        return self.env.action_space.sample()


    def choose_action(self, episode, step, state):
        if step == 0:
            self.action_stat = np.zeros(self.n_actions, dtype=np.int32)
        if step < self.launch_control:
            action = 3
        else:

            if state[0][0][4] >= state[0][1][4] and state[0][1][4] >= state[0][2][4]:
                action = 3
            
            elif state[0][0][3] >= state[0][0][4]:
                action = 2
            elif state[0][0][5] >= state[0][0][4]:
                action = 1
            
            elif state[0][0][2] >= state[0][0][4]:
                action = 2
            elif state[0][0][6] >= state[0][0][4]:
                action = 1
            
            elif state[0][0][1] >= state[0][0][4]:
                action = 2
            elif state[0][0][7] >= state[0][0][4]:
                action = 1
            
            elif state[0][0][0] >= state[0][0][4]:
                action = 2
            elif state[0][0][8] >= state[0][0][4]:
                action = 1
            
            elif state[0][0][4] < state[0][1][4] and state[0][1][4] < state[0][2][4] and state[0][2][4] - state[0][0][4] > 0.0429 and state[0][2][4] - state[0][1][4] > 0.028 and state[0][2][4] < 0.828 and state[0][0][4] > 0.5:
                action = 4
            
            elif state[0][0][4] >= max(state[0][0][3], state[0][0][5]):
                action = 3

            elif state[0][0][4] >= max(state[0][0][2], state[0][0][6]):
                action = 3
            
            else:
                print('RANDOM ACTION')
                action = self._get__random_action()

        self.action_stat[action] += 1
        self.action_stat_percentage = np.round(self.action_stat / np.sum(self.action_stat), 2)
        return action
