#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:14:34 2020

@author: anselme
"""

import sys
sys.path.insert(0, './utils')
import numpy as np
import matplotlib.pyplot as plt
import math
from cliffwalk import CliffWalk


# define MDP
env = CliffWalk(proba_succ=1)
S, A = env.Ns, env.Na
print(env.action_names)

# example of interaction between MDP and (random) policy
state = env.reset() # initial state
env.render()
t = 0
while t < 10:
    action = np.random.randint(0,A)
    next_state, reward, done, _ = env.step(action)
    env.render()
    
    if done:
        # terminal state reached, reset
        state = env.reset()
    t += 1

# QLEARNING
max_steps = 10000
epsilon = 0.1
rewards = np.zeros((max_steps))
t = 0
state = env.reset()
Q = np.zeros((S,A))
N = np.zeros((S,A), dtype=np.int) 
# QLEARNING
while t < max_steps:
    #######
    # action -> epsilon-greedy
    v = np.random.rand()
    if v < epsilon:
        action = np.random.randint(0, A)
    else:
        # action = np.argmax(Q[state])
        b = Q[state]
        idxs = np.flatnonzero(np.isclose(b, b.max()))
        idx = np.random.choice(idxs).item()
        action = env.actions[idx]
    #######
    next_state, reward, done, _ = env.step(action)
    rewards[t] = reward
    N[state, action] += 1

    #######
    # update Q
    td = reward + env.gamma * Q[next_state].max() * (1-done) - Q[state, action]
    alpha = 1. / np.sqrt(N[state, action])
    Q[state,action] = Q[state,action] + alpha * td
    #######

    state = next_state
    if done:
        state = env.reset()
    t = t + 1


plt.plot(rewards, 'o')
print('greedy policy wrt Q')
greedy_policy_ql = np.argmax(Q, axis=1)
env.render_policy(greedy_policy_ql)
plt.show()