import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from math import log
import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.95  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(21, input_dim=np.asarray(1), activation='relu'))
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(21, activation='relu'))
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

    def replay(self, batch_size, time, portfolio, state_space, action_space):
          minibatch = random.sample(self.memory, batch_size)
          for state, action, reward, next_state in minibatch:
              idx_state = np.argwhere(state_space==state)[0,0]
              # idx_action = np.argwhere(action_space==action)[0,0]
              target = (reward + self.gamma *
                        np.amax(self.model.predict(np.expand_dims(np.asarray(state), axis=0))[0]))
              target_f = self.model.predict(np.expand_dims(np.asarray(state), axis=0))
              print(f"state: {state_space[idx_state]} | action: {action_space[action]} | reward: {reward} | portfolio: {portfolio} | time {time}")
              print(f"before | target: {target} | target_f: {target_f}  | time: {time}")
              target_f[0][action] = target
              print(f"after | target: {target} | target_f: {target_f} | time: {time}")
              self.model.fit(np.expand_dims(np.asarray(state), axis=0), target_f, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon

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
        self.rho = np.random.choice(a=self.state_space, p=self.probabilities, size=50, replace=True)

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

    def get_next_action(self, q_table, state, action_space, epsilon, state_space):
        if np.random.random() < epsilon:
            action = np.random.choice(action_space)
            return np.argwhere(action_space == action)[0, 0]
        else:
          # print(f"state: {state} | state_idx = {np.argwhere(state_space==state)[0,0]}")
          state_idx=np.argwhere(state_space==state)[0,0]
          # print(f"return will be: {np.argmax(q_table[:, state_idx])}")
          return np.argmax(q_table[:, state_idx])


    def step(self, action, state, time, epsilon, gamma, alpha):
        # расчет reward и наблюдение нового состояния и действия
        reward = self.reward_function(self.portfolio)

        # определение следуюищего state & action
        next_state = self.rho[time]
        next_action = self.get_next_action(q_table=self.q_table, state=state, action_space=self.action_space,
                                           epsilon=epsilon, state_space=self.state_space)
        # обновление q_values
        # print(f"action: {action} | state: {state} | next_state: {next_state}")
        idx_state = np.argwhere(self.state_space==state)[0,0]
        idx_next_state = np.argwhere(self.state_space==next_state)[0,0]
        self.q_table[action, idx_state] = self.q_table[action, idx_state] * (1 - alpha) + alpha * (
                    reward + (gamma * np.max(self.q_table[:, idx_next_state])))
        # обновление значения портфеля
        self.portfolio = self.state_equation(portfolio=self.portfolio, action=self.action_space[next_action],
                                             rho=next_state)
        state = next_state
        action = next_action
        return state, action, reward

    def reset(self, portfolio):
      portfolio = 1000
      return self.rho[0], portfolio

if __name__ == "__main__":
    e = np.linspace(start=0, stop=1, num=6)
    probabilities = np.linspace(start=0.001, stop=0.05, num=len(e))
    probabilities = probabilities / np.sum(probabilities)
    action_space = np.linspace(start=-1, stop=1, num=21)
    env = Environment(action_space=action_space, state_space=e, probabilities=probabilities)
    print(env.state_size, env.action_size)
    EPISODES = 50
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for episode in range(EPISODES):
        state, env.portfolio = env.reset(env.portfolio)
        # state = np.reshape(state, [1, state_size])
        for time in range(len(env.rho)):
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
                agent.replay(batch_size, time=time, portfolio=env.portfolio, state_space=env.state_space, action_space=env.action_space)
        agent.update_epsilon()
# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95 # discount rate
#         self.epsilon = 1.0 # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()
#
#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#         optimizer=Adam(lr=self.learning_rate))
#         return model
#
#     def memorize(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0]) # returns action
#
#     def replay(self, batch_size, time):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = (reward + self.gamma *
#                 np.amax(self.model.predict(next_state)[0]))
#
#             target_f = self.model.predict(state)
#             print(f"before target: {target} | target_f: {target_f} | time: {time}")
#             target_f[0][action] = target # updating table of Q-value action
#             print(f"after target: {target} | target_f: {target_f} | time: {time}")
#
#             self.model.fit(state, target_f, epochs=1, verbose=1) # fitting weights of NEURAL NETWORK
#
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#     def load(self, name):
#         self.model.load_weights(name)
#
#     def save(self, name):
#         self.model.save_weights(name)


# if __name__ == "__main__":
#     EPISODES = 1000
#     env = gym.make('CartPole-v1')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     # agent.load("./save/cartpole-dqn.h5")
#     done = False
#     batch_size = 32
#     for e in range(EPISODES):
#         state = env.reset()
#         print(state[0])
#         state = np.reshape(state[0], [1, state_size])
#         for time in range(50):
#             # env.render()
#             action = agent.act(state)
#             next_state, reward, done, _, _ = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.memorize(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size, time=time)
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#             .format(e, EPISODES, time, agent.epsilon))
#             # if e % 10 == 0:
#             # agent.save("./save/cartpole-dqn.h5")