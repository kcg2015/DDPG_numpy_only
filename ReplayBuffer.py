#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 09:40:56 2017
Replay Buffer
@author: kyleguan
"""

from collections import deque
import random
import gym 
import numpy as np

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64



class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
        
        
        
if __name__ == '__main__':
    
    env = gym.make(ENV_NAME)
    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    for i in range(5):
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(100):

            if RENDER_ENV: 
                env.render()
            a = env.action_space.sample()
            s2, r, terminal, info = env.step(a)
           # replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, \
          #      terminal, np.reshape(s2, (actor.s_dim,)))
            replay_buffer.add(s, a, r, s2, terminal)
     
    batch = replay_buffer.getBatch(5)     
    print(replay_buffer.count())
    print(type(batch))
    print(len(batch))
    print(batch)       

    states = np.asarray([e[0] for e in batch])
    actions = np.asarray([e[1] for e in batch])
    rewards = np.asarray([e[2] for e in batch])
    new_states = np.asarray([e[3] for e in batch])
    dones = np.asarray([e[4] for e in batch])
    y_t = np.asarray([e[1] for e in batch])