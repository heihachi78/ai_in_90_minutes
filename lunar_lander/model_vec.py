import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from memory_vec import Memory



class BaseModel():

    def __init__() -> None:
        pass


    def _create_model(self) -> Model:
        return None


    def predict(self, state : np.ndarray) -> np.ndarray:
        return None


    def learn(self, mem : Memory, n_samples : int) -> None:
        pass


    def save(self) -> None:
        pass


    def load(self) -> None:
        pass


    def refresh_weights(self) -> None:
        pass


class ModelActorCritic(BaseModel):

    def __init__(self, n_input : int, n_output : int, n_neurons : list,
                 learning_rate : float, gamma : float, save_file : str, verbose : int, device : str) -> None:
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.actor : Model = self._create_model()
        self.target : Model = self._create_model()
        self.device = device
        self.save_file = 'weights_' + save_file + '_' + self.__class__.__name__ + '_' + self.device + '.weights.h5'


    def _create_model(self) -> Model:
        model = Sequential([
            Input(shape=(self.n_input,)),
            Dense(self.n_neurons[0], activation='relu'),
            Dense(self.n_neurons[1], activation='relu'),
            Dense(self.n_output, activation='linear', name='custom_model')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        #model.summary()

        return model


    def predict(self, states : np.ndarray) -> np.ndarray:
        return self.actor.predict(x=states, verbose=self.verbose)


    def learn(self, mem : Memory, n_samples : int, offset : int) -> None:
        if n_samples > mem.count:
            return

        states, actions, rewards, new_states, terminates = mem.sample(n_samples=n_samples, offset=offset)

        q_table = self.actor.predict(x=states, verbose=self.verbose)
        q_table_next_action = self.actor.predict(x=new_states, verbose=self.verbose)
        q_table_next = self.target.predict(x=new_states, verbose=self.verbose)


        batch_index = np.arange(n_samples, dtype=np.int32)
        action_values = np.array(list(range(self.n_output)), dtype=np.int8)
        action_indices = np.dot(actions, action_values)
        max_actions = np.argmax(q_table_next_action, axis=1)
        q_table[batch_index, action_indices] = rewards[batch_index]
        q_table[batch_index, action_indices] += self.gamma * q_table_next[batch_index, max_actions] * terminates[batch_index]

        self.actor.fit(x=states, y=q_table, verbose=self.verbose, batch_size=n_samples, epochs=1)


    def save(self) -> None:
        self.actor.save(self.save_file)
        print(f'weights saved to {self.save_file}')


    def load(self) -> None:
        try:
            self.actor = load_model(self.save_file)
            self.target = load_model(self.save_file)
            print(f'using weights {self.save_file}')
        except:
            print(f'cannot load weights {self.save_file}')


    def refresh_weights(self) -> None:
        self.target.set_weights(self.actor.get_weights())


class CustomModel(Model):

    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(fc1_dims, activation='relu')
        self.dense2 = Dense(fc2_dims, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)
        self.P = Dense(n_actions, activation='softmax')


    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q


    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A


class DuelingDeepQNetwork(BaseModel):

    def __init__(self, n_input : int, n_output : int, n_neurons : list,
                 learning_rate : float, gamma : float, save_file : str, verbose : int, device : str) -> None:
        self.n_input = n_input
        self.n_output = n_output
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.actor : CustomModel = self._create_model()
        self.target : CustomModel = self._create_model()
        self.device = device
        self.save_file = 'weights_' + save_file + '_' + self.__class__.__name__ + '_' + self.device + '.weights.h5'
        self.build = False
        self.predict(np.zeros((1, n_input), dtype=np.float32))


    def _create_model(self) -> Model:
        cm : CustomModel = CustomModel(fc1_dims=self.n_neurons[0], fc2_dims=self.n_neurons[1], n_actions=self.n_output)
        cm.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        cm.build(input_shape=(1, self.n_input))
        return cm


    def predict(self, states : np.ndarray) -> np.ndarray:
        actions = self.actor.advantage(states).numpy()
        if not self.build:
            _ = self.actor.call(states)
            _ = self.target.advantage(states)
            _ = self.target.call(states)
            self.build = True
        return actions


    def learn(self, mem : Memory, n_samples : int, offset : int) -> None:
        if n_samples > mem.count:
            return

        states, actions, rewards, new_states, terminates = mem.sample(n_samples=n_samples, offset=offset)

        q_table = self.actor(states).numpy()
        q_table_next_action = self.actor(new_states).numpy()
        q_table_next = self.target(new_states).numpy()

        batch_index = np.arange(n_samples, dtype=np.int32)
        action_values = np.array(list(range(self.n_output)), dtype=np.int8)
        action_indices = np.dot(actions, action_values)
        max_actions = np.argmax(q_table_next_action, axis=1)
        q_table[batch_index, action_indices] = rewards[batch_index]
        q_table[batch_index, action_indices] += self.gamma * q_table_next[batch_index, max_actions] * terminates[batch_index]

        self.actor.fit(x=states, y=q_table, verbose=self.verbose, batch_size=n_samples, epochs=1)


    def save(self) -> None:
        self.actor.save_weights(self.save_file)
        print(f'weights saved to {self.save_file}')


    def load(self) -> None:
        try:
            self.actor.load_weights(self.save_file)
            self.target.load_weights(self.save_file)
            print(f'using weights {self.save_file}')
        except:
            print(f'cannot load weights {self.save_file}')


    def refresh_weights(self) -> None:
        self.target.set_weights(self.actor.get_weights())