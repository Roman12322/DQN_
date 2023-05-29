import numpy as np
from math import log, pow
# import plotly.graph_objects as go
# from tqdm import tqdm
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=np.asarray(1), activation='relu'))
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(np.asarray(state), axis=0))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            print(state, action, reward, next_state)
            target = reward
            #     # if not done:
            target = (reward + self.gamma *
                      np.amax(self.model.predict(np.expand_dims(np.asarray(state), axis=0))[0]))
            target_f = self.model.predict(np.expand_dims(np.asarray(state), axis=0))
            target_f[0][action] = target
            self.model.fit(np.expand_dims(np.asarray(state), axis=0), target_f, epochs=1, verbose=0)
        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Environment:
    def __init__(self, action_space, state_space, probabilities):
        self.action_space = action_space
        self.state_space = state_space
        self.probabilities = probabilities
        self.action_size = len(action_space)
        self.state_size = len(state_space)
        self.q_table = np.ones([self.action_size, self.state_size]) * (-1000000)
        self.portfolio = 1000
        self.rho = np.random.choice(a=self.state_space, p=self.probabilities, size=500, replace=True)

    def reward_function(self, portfolio):
        if portfolio < 10 ** -17:
            return log(10 ** -17)
        elif portfolio > 10 ** 17:
            return log(10 ** 17)
        else:
            return log(portfolio)

    def state_equation(self, portfolio, action, rho):
        try:
            value = portfolio * (1 + (action * rho))
        except RuntimeWarning as e:
            print(f"{portfolio = }, {action = }, {rho = }")
            print(e)
        return value

    def get_next_action(self, q_table, state, action_space, epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice(action_space)
            return np.argwhere(action_space == action)[0, 0]
        else:
            return np.argmax(q_table[:, state])

    def get_next_state(self, state_space, next_state_value):
        return np.argwhere(state_space == next_state_value)[0, 0]

    def step(self, action, state, time, epsilon, gamma, alpha):
        # расчет reward и наблюдение нового состояния и действия
        reward = self.reward_function(self.portfolio)
        # определение следуюищего state & action
        next_state = self.get_next_state(state_space=self.state_space, next_state_value=self.rho[time])
        next_action = self.get_next_action(q_table=self.q_table, state=state, action_space=self.action_space,
                                           epsilon=epsilon)
        print(
            f"state: {state} | next_state: {next_state} | action: {action} | next_action: {next_action} | reward: {reward}")
        # обновление q_values
        self.q_table[action, state] = self.q_table[action, state] * (1 - alpha) + alpha * (
                    reward + (gamma * np.max(self.q_table[:, next_state])))
        # обновление значения портфеля
        self.portfolio = self.state_equation(portfolio=self.portfolio, action=self.action_space[next_action],
                                             rho=self.rho[next_state])
        state = next_state
        action = next_action
        return state, action, reward

    def reset(self, state_space):
        return np.where(state_space == state_space[0])[0][0]


EPISODES = 1000

if __name__ == "__main__":
    e = np.linspace(start=-0.5, stop=0, num=20)
    probabilities = np.linspace(start=0.001, stop=0.05, num=len(e))
    probabilities = probabilities / np.sum(probabilities)
    action_space = np.linspace(start=-1, stop=1, num=21)

    env = Environment(action_space=action_space, state_space=e, probabilities=probabilities)

    EPISODES = 10

    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    for episode in range(EPISODES):
        state = env.reset(env.state_space)
        # state = np.reshape(state, [1, state_size])
        for time in range(50):
            # env.render()
            action = agent.act(state)
            next_state, next_action, reward, = env.step(action=action, state=state, time=time,
                                                        epsilon=agent.epsilon, alpha=agent.learning_rate,
                                                        gamma=agent.gamma)
            # next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state)
            state = next_state
            # if done:
            print("episode: {}/{} | score: {} | e: {:.2} | memory: {}"
                  .format(episode, EPISODES, time, agent.epsilon, len(agent.memory)))
            # break
            if len(agent.memory) > batch_size:
                print(f"memory: {len(agent.memory)} | batch_size: {batch_size}")
                agent.replay(batch_size)











