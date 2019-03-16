import random
import os
import datetime
import numpy as np
from collections import deque
import tensorflow as tf


class DQNAgent:
    def __init__(self, action_size, state_size, batch_size=32, lr=0.000001, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, memory_size=10000, model_dir="."):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
	self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

        dt = datetime.datetime.now() 
        model_name = 'autobidding-dqn_{}.h5'.format(dt.strftime("%Y%m%d"))
        self.model_path = os.path.join(model_dir, model_name)

        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(15, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(300, activation='relu'))
        model.add(tf.keras.layers.Dense(200, activation='relu'))
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        opt = tf.keras.optimizers.RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=opt)

	return model

    def act(self, state, method=None):
        if method == 'random':
            return random.randrange(self.action_size)
        else:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                act_value = self.model.predict(state)
                #print(act_value)
                return np.argmax(act_value[0])
    

    def predict(self, state):
        return self.model.predict(state)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def store_transition(self, state, action, reward, done, next_state):
        observation = Observation(state, action, reward, next_state, done)
        self.memory.append(observation)

    def load(self, model_path):
        self.model_path = model_path
        print(self.model_path)
        if os.path.exists(model_path):
            print("loading model : {}".format(model_path))
            self.model.load_weights(model_path)

    def save(self):
        self.model.save_weights(self.model_path)


    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        batch_size = len(batch)
        states = np.array([o.state.to_array() for o in batch])
        next_states = np.array([o.next_state.to_array() for o in batch])
 
        p = self.target_model.predict(states)
        next_p = self.predict(next_states)

        targets = p

        for i in range(batch_size):
            # obs : state, action, action_idx, reward, done, next_state
            obs = batch[i]
            reward = obs.reward
            action_id = obs.action
            done = obs.done
            if done:
                targets[i][action_id] = reward
            else:
                targets[i][action_id] = reward + self.gamma*np.amax(next_predict_v[i])

        self.model.fit(states, targets, epochs=1, verbose=1)


class Observation:
    def __init__(self, state, action_id, reward, next_state, done):
        self.state = state
        self.action = action_id
        self.reward = reward
        self.next_state = next_state
        self.done = done
