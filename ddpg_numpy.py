#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This version test the hand-coded neural net for both
actor and critic netorks.

@author: kyleguan
"""
# import python packages
import numpy as np
import gym
import actor_net
import critic_net
from ReplayBuffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 40
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# Parameters for neural net
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
L2_REG_SCALE = 0
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

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
RANDOM_SEED = 11111
# Size of replay buffer
ACTION_BOUND=2


 

if __name__ == '__main__':        
    
    env = gym.make(ENV_NAME).env
    np.random.seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # Create actor and critic nets
    actor = ActorNet(state_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, action_dim)
    critic = CriticNet(state_dim, action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS,
                        HIDDEN2_UNITS, action_dim)
    
    # Initialize replay buffer
    
    buff = ReplayBuffer(BUFFER_SIZE)      
    
    step=0
    reward_result=[]

    for i in range(MAX_EPISODES): 

        s_t = env.reset()
        total_reward = 0.
        for j in range(MAX_EP_STEPS):
            loss=0;
            loss2 = 0;

            if RENDER_ENV: 
                env.render()
            # Select action according to the cuurent policy and exploration noise    
            # add noise in the form of 1./(1.+i+j), decaying over episodes and
            # steps, otherwise a_t will be the same, since s is fixed per episode.
            a_t = actor.predict(np.reshape(s_t,(1,3)), ACTION_BOUND, target=False)+1./(1.+i+j)
            
            # Execute action a_t and observe reward r_t and new state s_{t+1}
            s_t_1, r_t, done, info = env.step(a_t[0])
            
            # Store transition in replay buffer
            buff.add(s_t, a_t[0], r_t, s_t_1, done)
            
            # If the no. of experiences (episodes) is larger than the mini batch size
            if buff.count() > MINIBATCH_SIZE:
                # Sample a random batch 
                batch = buff.getBatch(MINIBATCH_SIZE)
                states_t = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                states_t_1 = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                # Setup y_is for updating critic
                y=np.zeros((len(batch), action_dim))
                a_tgt=actor.predict(states_t_1, ACTION_BOUND, target=True)
                Q_tgt = critic.predict(states_t_1, a_tgt,target=True)
                
                for i in range(len(batch)):
                    if dones[i]:
                        y[i] = rewards[i]
                    else:
                        y[i] = rewards[i] + GAMMA*Q_tgt[i]    
                # Update critic by minimizing the loss
                loss += critic.train(states_t, actions, y)
                # Update actor using sampled policy gradient
                a_for_dQ_da=actor.predict(states_t, ACTION_BOUND, target=False)
                dQ_da = critic.evaluate_action_gradient(states_t,a_for_dQ_da)
                actor.train(states_t, dQ_da, ACTION_BOUND)
                
                # Update target networks
                actor.train_target(TAU)
                critic.train_target(TAU)
                
            s_t = s_t_1
            total_reward += r_t    
            
            step += 1
            if done:
                "Done!"
                break
        reward_result.append(total_reward)
        print("TOTAL REWARD @ " + str(i) +"-th Episode:" + str(total_reward))
        print("Total Step: " + str(step))
        print("")       