
# LIBRARIES

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





## Please run the QNet chunk before the the single run of DQN, mutliple run of DQN and the DDQN

class QNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        #
        Q = self.fc1(state)
        Q = self.relu(Q)
        
        Q = self.fc2(Q)
        Q = self.relu(Q)
        
        Q = self.fc3(Q)
        # ====================================================
        return Q

    def select_greedyaction(self, state):
        with torch.no_grad():
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            #
            # detach in the gradient calculation and evaluation our model:
            with torch.no_grad():
                action_index = torch.argmax(self.forward(state),1)
            # ====================================================
        return action_index.item()








### Please run this script before running all Networks ( QUESTIONs)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition.
            sample is a tuple (state, next_state, action, reward, done)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        samples = random.sample(self.memory, batch_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)

def eval_dqn(env, qnet, n_sim=5):
    """
    Monte Carlo evaluation of DQN agent
    """
    rewards = np.zeros(n_sim)
    copy_env = deepcopy(env) # Important!
    # Loop over number of simulations
    for sim in range(n_sim):
        state = copy_env.reset()
        done = False
        while not done:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = qnet.select_greedyaction(tensor_state)
            next_state, reward, done, _ = copy_env.step(action)
            # update sum of rewards
            rewards[sim] += reward
            state = next_state
    return rewards





######### GBOBAL CONSTANT 

# please run this chunk before all questions

# Discount factor
GAMMA = 0.99
EVAL_EVERY = 2

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 30000
# Update target net every ... episodes
UPDATE_TARGET_EVERY = 20

# Initial value of epsilon
EPSILON_START = 1.0
# Parameter to decrease epsilon
DECREASE_EPSILON = 200
# Minimum value of epislon
EPSILON_MIN = 0.05

# Number of training episodes
N_EPISODES = 250

# Learning rate
LEARNING_RATE = 1e-3

env = gym.make('CartPole-v0')









######## FIRST QUESTION : single run of DQN 

# initialize replay buffer

replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# create network and target network
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
# ====================================================
# YOUR IMPLEMENTATION HERE 
# Define networks
#
q_net = QNet(obs_size, n_actions).to(device)

target_net = QNet(obs_size, n_actions).to(device)
# ====================================================

# objective and optimizer
optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)

# Algorithm
state = env.reset()
epsilon = EPSILON_START
ep = 0
total_time = 0
learn_steps = 0
episode_reward = 0

episode_rewards = np.zeros((N_EPISODES, 3))
while ep < N_EPISODES:
    # ====================================================
    # YOUR IMPLEMENTATION HERE 
    # sample epsilon-greedy action
    #
    p = random.random()
    if p < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = q_net.select_greedyaction(tensor_state)
        
    # ====================================================

    next_state, reward, done, _ = env.step(action)
    episode_reward += reward
    total_time += 1
    
    # ====================================================
    # YOUR IMPLEMENTATION HERE 
    # add sample to buffer
    #
    replay_buffer.push((state, next_state, action, reward, done))
    # ====================================================


    if len(replay_buffer) > BATCH_SIZE:
        learn_steps += 1
        # UPDATE MODEL
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # get batch
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)
        # ====================================================


        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

        with torch.no_grad():
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # build target (recall that we conseder the Q function
            # in the next state only if not terminal, ie. done != 1)
            # (1- done) * value_next
            #
            # targets = ...
            values_next  = target_net(batch_next_state).max(1)[0].unsqueeze(1)
            targets = values_next*(1-batch_done)
            targets = batch_reward + GAMMA*targets
            # ====================================================

        values = q_net(batch_state).gather(1, batch_action.long())

        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # compute loss and update model (loss and optimizer)
        optimizer.zero_grad()
        
        criterion = torch.nn.MSELoss()
        loss = criterion(targets, values)
        loss.backward()
        optimizer.step()
        # ====================================================

        if epsilon > EPSILON_MIN:
            epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON
    


    # ====================================================
    # YOUR IMPLEMENTATION HERE 
    # update target network
    if learn_steps % UPDATE_TARGET_EVERY == 0:
        target_net.load_state_dict(q_net.state_dict())
    # ====================================================

    state = next_state
    if done:
        mean_rewards = -1
        if (ep+1) % EVAL_EVERY == 0:
            # evaluate current policy
            rewards = eval_dqn(env, q_net)
            mean_rewards = np.mean(rewards)
            print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
            # if np.mean(rewards) >= REWARD_THRESHOLD:
            #     break
        
        episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
        state = env.reset()
        ep += 1
        episode_reward = 0

###################################################################
# VISUALIZATION
###################################################################
for episode in range(3):
    done = False
    state = env.reset()
    env.render()
    while not done:
        tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = q_net.select_greedyaction(tensor_state)
        state, reward, done, info = env.step(action)
        env.render()

plt.figure()
plt.title('Performance over learning')
plt.plot(episode_rewards[:,0], episode_rewards[:,1])
plt.xlabel('time steps')
plt.ylabel('total reward')

plt.figure()
plt.title('Performance on Test Env')
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(episode_rewards[xv, 0], episode_rewards[xv, 2], ':o')
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.show()

#### results in DQN1.txt
file_out = open('DQN1.txt', 'w')
file_out.write("times, rtrain, rtest\n")
for i in range(episode_rewards.shape[0]):
    file_out.write('%d, %f, %f\n'% (episode_rewards[i][0].astype(int), episode_rewards[i][1], episode_rewards[i][2]))
file_out.close()

import pandas as pd 
p2 = pd.read_table('DQN1.txt', sep = ',')
p2 = p2.to_numpy()



######## SECOND CODIND QUESTION : Multiple trial of DQN.

# we use the function multi_rep below

# if there is a warming due to the use of deepcopy, please clean the variable explorer
# run once again the chunk of libraries, the chunk of definition of the QNet, the RepalyBuffer class and 
# the function eval_dqn. 
# Please run also the GBOBAL CONSTANT chunk 

def multi_rep(trial):
    episode_rewards_trial  = np.zeros((trial, N_EPISODES, 3))
    
    # boucle 
    for t in range(trial):
        
        print("trial %d"%(t+1))
            # initialize replay buffer
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        
        # create network and target network
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # Define networks
        #
        q_net = QNet(obs_size, n_actions).to(device)
        
        target_net = QNet(obs_size, n_actions).to(device)
        # ====================================================
        
        # objective and optimizer
        optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)
        
        # Algorithm
        state = env.reset()
        epsilon = EPSILON_START
        ep = 0
        total_time = 0
        learn_steps = 0
        episode_reward = 0
        
        episode_rewards = np.zeros((N_EPISODES, 3))
        while ep < N_EPISODES:
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # sample epsilon-greedy action
            #
            p = random.random()
            if p < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_net.select_greedyaction(tensor_state)
                
            # ====================================================
        
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_time += 1
            
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # add sample to buffer
            #
            replay_buffer.push((state, next_state, action, reward, done))
            # ====================================================
        
        
            if len(replay_buffer) > BATCH_SIZE:
                learn_steps += 1
                # UPDATE MODEL
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)
                # ====================================================
        
        
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
        
                with torch.no_grad():
                    # ====================================================
                    # YOUR IMPLEMENTATION HERE 
                    # build target (recall that we conseder the Q function
                    # in the next state only if not terminal, ie. done != 1)
                    # (1- done) * value_next
                    #
                    # targets = ...
                    
                    
                    values_next  = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    targets = values_next*(1-batch_done)
                    targets = batch_reward + GAMMA*targets
                    # ====================================================
        
                values = q_net(batch_state).gather(1, batch_action.long())
        
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # compute loss and update model (loss and optimizer)
                optimizer.zero_grad()
                
                criterion = torch.nn.MSELoss()
                loss = criterion(targets, values)
                loss.backward()
                optimizer.step()
                # ====================================================
        
                if epsilon > EPSILON_MIN:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON
            
        
        
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # update target network
            if learn_steps % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # ====================================================
        
            state = next_state
            if done:
                mean_rewards = -1
                if (ep+1) % EVAL_EVERY == 0:
                    # evaluate current policy
                    rewards = eval_dqn(env, q_net)
                    mean_rewards = np.mean(rewards)
                    print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
                    # if np.mean(rewards) >= REWARD_THRESHOLD:
                    #     break
                
                episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
                state = env.reset()
                ep += 1
                episode_reward = 0

        episode_rewards_trial[t,:,:] = episode_rewards
    return episode_rewards_trial


# save and plot of results : 
    
trials = 20
result_DQN2 = multi_rep(trials)



# we will plot the first timestep arbitrary 
time_steps = result_DQN2[0,:,0]
avg_episode_reward = result_DQN2[:,:,1].mean(axis = 0)
std_episode_reward = result_DQN2[:,:,1].std(axis = 0)

avg_mean_reward = result_DQN2[:,:,2].mean(axis = 0)
std_mean_reward = result_DQN2[:,:,2].std(axis = 0)

DQN_save = np.vstack( (time_steps, avg_episode_reward,std_episode_reward, avg_mean_reward, std_mean_reward)).T



### the file saved is DQN20
file_out = open('DQN20.txt', 'w')
file_out.write("times, avg_rtrain, std_train, avg_test, std_test,\n")
for i in range(DQN_save.shape[0]):
    file_out.write('%d, %f, %f, %f, %f\n'% (DQN_save[i][0].astype(int), DQN_save[i][1], DQN_save[i][2],DQN_save[i][3], DQN_save[i][4]))
file_out.close()

import pandas as pd 
p2 = pd.read_table('DQN20.txt', sep = ',')
p2 = p2.to_numpy()


# plots : 
plt.figure()
plt.title('Performance over learning for %d trials'%trials)
plt.plot(p2[:,0], p2[:,1], label = 'DQN')
st = 2*(p2[:,2]/math.sqrt(trials)) # standard deviation over the number of trial 
plt.fill_between(p2[:,0], p2[:,1]-st, p2[:,1] +st, color = 'tab:blue', alpha = 0.3)
plt.xlabel('time steps')
plt.ylabel('total reward')
plt.legend()


plt.figure()
plt.title('Performance on Test Env for %d trial'%trials)
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(p2[xv, 0], p2[xv, 3], ':o', label = 'DQN')
st =  2*(p2[xv,4]/math.sqrt(trials))
plt.fill_between(p2[xv,0], p2[xv,3]- st, p2[xv,3]+ st, color = 'tab:blue', alpha = 0.2)
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.legend()
plt.show()









###### THIRD QUESTION:     Double DQNN

# this chunk depends on the LIBRARIES , QNet class, ReplayBuffer, eval_dqn and the initialisation of 
# GLOBAL CONSTANT.

# if there is problem with deepcopy, please clean your variable explorer , run once again these chunk

def multi_rep_DDQN(trial):
    episode_rewards_trial  = np.zeros((trial, N_EPISODES, 3))
    
    # boucle 
    for t in range(trial):
        
        print("trial %d"%(t+1))
            # initialize replay buffer
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        
        # create network and target network
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # Define networks
        #
        q_net = QNet(obs_size, n_actions).to(device)
        
        target_net = QNet(obs_size, n_actions).to(device)
        # ====================================================
        
        # objective and optimizer
        optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)
        
        # Algorithm
        state = env.reset()
        epsilon = EPSILON_START
        ep = 0
        total_time = 0
        learn_steps = 0
        episode_reward = 0
        
        episode_rewards = np.zeros((N_EPISODES, 3))
        while ep < N_EPISODES:
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # sample epsilon-greedy action
            #
            p = random.random()
            if p < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_net.select_greedyaction(tensor_state)
                
            # ====================================================
        
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_time += 1
            
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # add sample to buffer
            #
            replay_buffer.push((state, next_state, action, reward, done))
            # ====================================================
        
        
            if len(replay_buffer) > BATCH_SIZE:
                learn_steps += 1
                # UPDATE MODEL
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)
                # ====================================================
        
        
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
        
                with torch.no_grad():
                    # ====================================================
                    # YOUR IMPLEMENTATION HERE 
                    # build target (recall that we conseder the Q function
                    # in the next state only if not terminal, ie. done != 1)
                    # (1- done) * value_next
                    #
                    # targets = ...
                    
                    greedy_next_qnet = torch.argmax(q_net(batch_next_state),1).unsqueeze(1)
                    values_next = target_net(batch_next_state).gather(1, greedy_next_qnet.long())
                    
                    targets = values_next*(1-batch_done)
                    targets = batch_reward + GAMMA*targets
                    
                    # ====================================================
        
                values = q_net(batch_state).gather(1, batch_action.long())
        
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # compute loss and update model (loss and optimizer)
                optimizer.zero_grad()
                
                criterion = torch.nn.MSELoss()
                loss = criterion(targets, values)
                loss.backward()
                optimizer.step()
                # ====================================================
        
                if epsilon > EPSILON_MIN:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON
            
        
        
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # update target network
            if learn_steps % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # ====================================================
        
            state = next_state
            if done:
                mean_rewards = -1
                if (ep+1) % EVAL_EVERY == 0:
                    # evaluate current policy
                    rewards = eval_dqn(env, q_net)
                    mean_rewards = np.mean(rewards)
                    print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
                    # if np.mean(rewards) >= REWARD_THRESHOLD:
                    #     break
                
                episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
                state = env.reset()
                ep += 1
                episode_reward = 0

        episode_rewards_trial[t,:,:] = episode_rewards
    return episode_rewards_trial


# plot and save file for DDQN :
    
trials = 20


result_DDQN2 = multi_rep_DDQN(trials)

# save and plot of results : 

# we will plot the first timestep arbitrary 
time_steps = result_DDQN2[0,:,0]
avg_episode_reward = result_DDQN2[:,:,1].mean(axis = 0)
std_episode_reward = result_DDQN2[:,:,1].std(axis = 0)

avg_mean_reward = result_DDQN2[:,:,2].mean(axis = 0)
std_mean_reward = result_DDQN2[:,:,2].std(axis = 0)

DDQN_save = np.vstack( (time_steps, avg_episode_reward,std_episode_reward, avg_mean_reward, std_mean_reward)).T



### the file saved is DDQN20
file_out = open('DDQN20.txt', 'w')
file_out.write("times, avg_rtrain, std_train, avg_test, std_test,\n")
for i in range(DDQN_save.shape[0]):
    file_out.write('%d, %f, %f, %f, %f\n'% (DDQN_save[i][0].astype(int), DDQN_save[i][1], DDQN_save[i][2],DDQN_save[i][3], DDQN_save[i][4]))
file_out.close()



import pandas as pd 
p2 = pd.read_table('DDQN20.txt', sep = ',')
p2 = p2.to_numpy()


# plot : 
plt.figure()
plt.title('Performance over learning for %d trials'%trials)
plt.plot(p2[:,0], p2[:,1], label = 'DDQN',color = 'r')
st = 2*(p2[:,2]/math.sqrt(trials)) # standard deviation over the number of trial 
plt.fill_between(p2[:,0], p2[:,1]- st, p2[:,1]+st, color = 'tab:red', alpha = 0.2)
plt.xlabel('time steps')
plt.ylabel('total reward')
plt.legend()


plt.figure()
plt.title('Performance on Test Env for %d trial'%trials)
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(p2[xv, 0], p2[xv, 3], ':o', label = 'DDQN',color = 'r')
st =  2*(p2[xv,4]/math.sqrt(trials))
plt.fill_between(p2[xv,0], p2[xv,3]-st, p2[xv,3]+st, color = 'tab:red', alpha = 0.2)
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.legend()
plt.show()








##### QUESTION FOUR : Dueling DDQN

# this part uses LIBRARIES, ReplayBuffer Class, eval_qnet, GLOBAL CONSTANT 
# so please run once again these chunk if necessary

class DueNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DueNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        
        self.fc2_v = nn.Linear(64, 256)
        self.fc2_a = nn.Linear(64, 256)
        
        self.fc3_v = nn.Linear(256, 1)
        self.fc3_a = nn.Linear(256, n_actions)
        
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        
        # left branch 
        x_v = self.fc2_v(x)
        x_v = self.relu(x_v)
        x_v = self.fc3_v(x_v)
        
        # right branch 
        x_a = self.fc2_a(x)
        x_a = self.relu(x_a)
        x_a = self.fc3_a(x_a)
        
        return x_v + x_a - torch.mean(x_a, dim =1, keepdim= True)
    def select_greedyaction(self, state):
        with torch.no_grad():
            action_index = torch.argmax(self.forward(state),1)
        return action_index.item()
    

 

    
def multi_rep_DDueNet(trial):
    episode_rewards_trial  = np.zeros((trial, N_EPISODES, 3))
    
    # boucle 
    for t in range(trial):
        
        print("trial %d"%(t+1))
            # initialize replay buffer
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        
        # create network and target network
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # Define networks
        #
        q_net = DueNet(obs_size, n_actions).to(device)
        
        target_net = DueNet(obs_size, n_actions).to(device)
        # ====================================================
        
        # objective and optimizer
        optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)
        
        # Algorithm
        state = env.reset()
        epsilon = EPSILON_START
        ep = 0
        total_time = 0
        learn_steps = 0
        episode_reward = 0
        
        episode_rewards = np.zeros((N_EPISODES, 3))
        while ep < N_EPISODES:
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # sample epsilon-greedy action
            #
            p = random.random()
            if p < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_net.select_greedyaction(tensor_state)
                
            # ====================================================
        
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_time += 1
            
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # add sample to buffer
            #
            replay_buffer.push((state, next_state, action, reward, done))
            # ====================================================
        
        
            if len(replay_buffer) > BATCH_SIZE:
                learn_steps += 1
                # UPDATE MODEL
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)
                # ====================================================
        
        
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
        
                with torch.no_grad():
                    # ====================================================
                    # YOUR IMPLEMENTATION HERE 
                    # build target (recall that we conseder the Q function
                    # in the next state only if not terminal, ie. done != 1)
                    # (1- done) * value_next
                    #
                    # targets = ...
                    
                    greedy_next_qnet = torch.argmax(q_net(batch_next_state),1).unsqueeze(1)
                    values_next = target_net(batch_next_state).gather(1, greedy_next_qnet.long())
                    
                    targets = values_next*(1-batch_done)
                    targets = batch_reward + GAMMA*targets   
                    # ====================================================
        
                values = q_net(batch_state).gather(1, batch_action.long())
        
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # compute loss and update model (loss and optimizer)
                optimizer.zero_grad()
                
                criterion = torch.nn.MSELoss()
                loss = criterion(targets, values)
                loss.backward()
                optimizer.step()
                # ====================================================
        
                if epsilon > EPSILON_MIN:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON
            
        
        
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # update target network
            if learn_steps % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # ====================================================
        
            state = next_state
            if done:
                mean_rewards = -1
                if (ep+1) % EVAL_EVERY == 0:
                    # evaluate current policy
                    rewards = eval_dqn(env, q_net)
                    mean_rewards = np.mean(rewards)
                    print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
                    # if np.mean(rewards) >= REWARD_THRESHOLD:
                    #     break
                
                episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
                state = env.reset()
                ep += 1
                episode_reward = 0

        episode_rewards_trial[t,:,:] = episode_rewards
    return episode_rewards_trial


# save and plot of results : 
# plot and save file for DueDDQN :
    
trials = 20
result_DueDDQN2 = multi_rep_DDueNet(trials)

# save and plot of results 

# we will plot the first timestep arbitrary 
time_steps = result_DueDDQN2[0,:,0]
avg_episode_reward = result_DueDDQN2[:,:,1].mean(axis = 0)
std_episode_reward = result_DueDDQN2[:,:,1].std(axis = 0)

avg_mean_reward = result_DueDDQN2[:,:,2].mean(axis = 0)
std_mean_reward = result_DueDDQN2[:,:,2].std(axis = 0)

DueDDQN_save = np.vstack( (time_steps, avg_episode_reward,std_episode_reward, avg_mean_reward, std_mean_reward)).T



### the file saved is DueDQN20
file_out = open('DueDDQN20.txt', 'w')
file_out.write("times, avg_rtrain, std_train, avg_test, std_test,\n")
for i in range(DueDDQN_save.shape[0]):
    file_out.write('%d, %f, %f, %f, %f\n'% (DueDDQN_save[i][0].astype(int), DueDDQN_save[i][1], DueDDQN_save[i][2],DueDDQN_save[i][3], DueDDQN_save[i][4]))
file_out.close()

import pandas as pd 
p2 = pd.read_table('DueDDQN20.txt', sep = ',')
p2 = p2.to_numpy()


# plot : 
plt.figure()
plt.title('Performance over learning for %d trials'%trials)
plt.plot(p2[:,0], p2[:,1], label = 'DueDDQN',color = 'yellow')
st = 2*(p2[:,2]/math.sqrt(trials))
plt.fill_between(p2[:,0], p2[:,1]- st, p2[:,1]+ st, color = 'tab:olive', alpha = 0.2)
plt.xlabel('time steps')
plt.ylabel('total reward')
plt.legend()


plt.figure()
plt.title('Performance on Test Env for %d trial'%trials)
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(p2[xv, 0], p2[xv, 3], ':o', label = 'DueDDQN', color = 'yellow')
st =  2*(p2[xv,4]/math.sqrt(trials))
plt.fill_between(p2[xv,0], p2[xv,3]- st, p2[xv,3]+ st, color = 'tab:olive', alpha = 0.3)
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.legend()
plt.show()
    
 

 
    
 
    
 
    

#### OPTIONAL : Standard Dueling DQN
def multi_rep_DueNet(trial):
    episode_rewards_trial  = np.zeros((trial, N_EPISODES, 3))
    
    # boucle 
    for t in range(trial):
        
        print("trial %d"%(t+1))
            # initialize replay buffer
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        
        # create network and target network
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        # ====================================================
        # YOUR IMPLEMENTATION HERE 
        # Define networks
        #
        q_net = DueNet(obs_size, n_actions).to(device)
        
        target_net = DueNet(obs_size, n_actions).to(device)
        # ====================================================
        
        # objective and optimizer
        optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)
        
        # Algorithm
        state = env.reset()
        epsilon = EPSILON_START
        ep = 0
        total_time = 0
        learn_steps = 0
        episode_reward = 0
        
        episode_rewards = np.zeros((N_EPISODES, 3))
        while ep < N_EPISODES:
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # sample epsilon-greedy action
            #
            p = random.random()
            if p < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_net.select_greedyaction(tensor_state)
                
            # ====================================================
        
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_time += 1
            
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # add sample to buffer
            #
            replay_buffer.push((state, next_state, action, reward, done))
            # ====================================================
        
        
            if len(replay_buffer) > BATCH_SIZE:
                learn_steps += 1
                # UPDATE MODEL
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)
                # ====================================================
        
        
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
        
                with torch.no_grad():
                    # ====================================================
                    # YOUR IMPLEMENTATION HERE 
                    # build target (recall that we conseder the Q function
                    # in the next state only if not terminal, ie. done != 1)
                    # (1- done) * value_next
                    #
                    # targets = ...
                    
                    values_next  = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    targets = values_next*(1-batch_done)
                    targets = batch_reward + GAMMA*targets
                    # ====================================================
        
                values = q_net(batch_state).gather(1, batch_action.long())
        
                # ====================================================
                # YOUR IMPLEMENTATION HERE 
                # compute loss and update model (loss and optimizer)
                optimizer.zero_grad()
                
                criterion = torch.nn.MSELoss()
                loss = criterion(targets, values)
                loss.backward()
                optimizer.step()
                # ====================================================
        
                if epsilon > EPSILON_MIN:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON
            
        
        
            # ====================================================
            # YOUR IMPLEMENTATION HERE 
            # update target network
            if learn_steps % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # ====================================================
        
            state = next_state
            if done:
                mean_rewards = -1
                if (ep+1) % EVAL_EVERY == 0:
                    # evaluate current policy
                    rewards = eval_dqn(env, q_net)
                    mean_rewards = np.mean(rewards)
                    print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
                    # if np.mean(rewards) >= REWARD_THRESHOLD:
                    #     break
                
                episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
                state = env.reset()
                ep += 1
                episode_reward = 0

        episode_rewards_trial[t,:,:] = episode_rewards
    return episode_rewards_trial


# save and plot of results : 
# plot and save file for DueDQN :
    
trials = 20
result_DueDQN2 = multi_rep_DueNet(trials)

# save and plot of results 

# we will plot the first timestep arbitrary 
time_steps = result_DueDQN2[0,:,0]
avg_episode_reward = result_DueDQN2[:,:,1].mean(axis = 0)
std_episode_reward = result_DueDQN2[:,:,1].std(axis = 0)

avg_mean_reward = result_DueDQN2[:,:,2].mean(axis = 0)
std_mean_reward = result_DueDQN2[:,:,2].std(axis = 0)

DueDQN_save = np.vstack( (time_steps, avg_episode_reward,std_episode_reward, avg_mean_reward, std_mean_reward)).T



### the file saved is DueDQN20
file_out = open('DueDQN20.txt', 'w')
file_out.write("times, avg_rtrain, std_train, avg_test, std_test,\n")
for i in range(DueDQN_save.shape[0]):
    file_out.write('%d, %f, %f, %f, %f\n'% (DueDQN_save[i][0].astype(int), DueDQN_save[i][1], DueDQN_save[i][2],DueDQN_save[i][3], DueDQN_save[i][4]))
file_out.close()

import pandas as pd 
p2 = pd.read_table('DueDQN20.txt', sep = ',')
p2 = p2.to_numpy()


# plot : 
plt.figure()
plt.title('Performance over learning for %d trials'%trials)
plt.plot(p2[:,0], p2[:,1], label = 'DueDQN',color = 'g')
st = 2*(p2[:,2]/math.sqrt(trials))
plt.fill_between(p2[:,0], p2[:,1]- st, p2[:,1]+ st, color = 'tab:green', alpha = 0.3)
plt.xlabel('time steps')
plt.ylabel('total reward')
plt.legend()


plt.figure()
plt.title('Performance on Test Env for %d trial'%trials)
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(p2[xv, 0], p2[xv, 3], ':o', label = 'DueDQN', color = 'g')
st =  2*(p2[xv,4]/math.sqrt(trials))
plt.fill_between(p2[xv,0], p2[xv,3]- st, p2[xv,3]+ st, color = 'tab:green', alpha = 0.3)
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.legend()
plt.show()



###### Comparison of the performance of DQN, DDQN, DueDQN

# import all saves :: 
import pandas as pd 
DQN20 = pd.read_table('DQN20.txt', sep = ',').to_numpy()
DDQN20 = pd.read_table('DDQN20.txt', sep = ',').to_numpy()
DueDDQN20 = pd.read_table('DueDDQN20.txt', sep = ',').to_numpy()


## Visualisation 

plt.figure()
plt.title('Performance over learning for %d trials'%trials)

plt.plot(DQN20 [:,0], DQN20[:,1], label = 'DQN',color = 'b')
plt.plot(DDQN20 [:,0], DDQN20[:,1], label = 'DDQN',color = 'r')
plt.plot(DueDDQN20[:,0], DueDDQN20[:,1], label = 'DueDDQN',color = 'yellow')
plt.xlabel('time steps')
plt.ylabel('total reward')
plt.legend()
plt.show()



plt.figure()
plt.title('Performance on Test Env for %d trial'%trials)
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.plot(DQN20[xv, 0], DQN20[xv, 3], ':o', label = 'DQN', color = 'b')
plt.plot(DDQN20[xv, 0], DDQN20[xv, 3], ':o', label = 'DDQN', color = 'r')
plt.plot(DueDDQN20[xv, 0], DueDDQN20[xv, 3], ':o', label = 'DueDDQN', color = 'yellow')
plt.xlabel('time steps')
plt.ylabel('expected total reward (greedy policy)')
plt.legend()
plt.show()

#### END 