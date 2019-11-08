import numpy as np
import gym
import random
import time

env = gym.make('FrozenLake-v0')
#get state ann action space size
actionSpaceSize = env.action_space.n
stateSpaceSize = env.observation_space.n
episodes = 1000
steps_per_episode = 100
exploration_rate = 1
exploration_rate_decay = 0.01
exploration_rate_max = 1
exploration_rate_min = 0.01
learning_rate = 0.001
discount_rate = 0.99

#initialize q table
q_table = np.zeros((stateSpaceSize,actionSpaceSize)) 
print(q_table)
all_rewards = []

for episode in range(episodes):
    #env.render()
    state = env.reset() # reset environment after each episode
    done = False #if done = true at episode end then...
    episode_reward = 0

    for i in range(steps_per_episode):
        exploration_threshold = random.uniform(0,1)
        if exploration_threshold>exploration_rate:
           # print('exploit')
            #exploit
            action = np.argmax(q_table[state,:])
        else:
            #randomly eplore and get environment information
            action = env.action_space.sample()
            #print('explore')
        new_state,reward,done,info = env.step(action)
        
        #update q table with learned q value with bellman equation
        q_table[state,action] = (1-learning_rate)*q_table[state,action]+learning_rate*(reward+discount_rate*np.max(q_table[new_state,action]))
    
        state = new_state
        episode_reward += reward
        #print( q_table[state,action],episode_reward) 
        
        if done == True:
            break # and reste environment
        #decay exploration rate with learning experience
       
        #exploration rate decay
        all_rewards.append(episode_reward)
        exploration_rate = exploration_rate_min+(exploration_rate_max-exploration_rate_min)*np.exp(-exploration_rate_decay*episode)
'''
rewards_per_thosand_episodes = np.split(np.array(all_rewards),episodes/100)
count = 1000

print("********Average reward per 10 thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
    '''
        
        
print(q_table)