import numpy as np
import gymnasium as gym
from memory_vec import Memory
from model_vec import BaseModel



class Agent():

    def __init__(self, envs : gym.vector.VectorEnv, memory : Memory, model : BaseModel, n_samples : int, eps : float, eps_dec : float, eps_min : float) -> None:
        self.envs = envs
        self.memory = memory
        self.model = model
        self.n_samples = n_samples
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min


    def act(self, states : np.ndarray, learning : bool) -> int:
        if learning:
            if np.random.rand() < self.eps:
                actions = self.envs.action_space.sample()
            else:
                pred = self.model.predict(states=states)
                actions = np.argmax(pred, axis=1)
            if self.eps > self.eps_min:
                self.eps *= self.eps_dec
                if self.eps_min > self.eps:
                    self.eps = self.eps_min
        else:
            pred = self.model.predict(states=states)
            actions = np.argmax(pred, axis=1)
        return actions


    def learn(self) -> None:
        self.model.learn(mem=self.memory, n_samples=self.n_samples)


    def remember(self, state : np.ndarray, action : int, reward : float, new_state : np.ndarray, terminate : bool) -> None:
        self.memory.store(state=state, action=action, reward=reward, new_state=new_state, terminate=terminate)
