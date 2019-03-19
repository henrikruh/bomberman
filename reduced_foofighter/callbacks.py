
import numpy as np
import matplotlib.pylab as plt
from random import shuffle
from time import time, sleep
from collections import deque
import os

from settings import s

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from .dqn import shallowQN
#from .tensorboard import Tensorboard

def get_state(self):
    
    # get arena 
    arena = self.game_state['arena']

    # get agent position
    x, y, _, bombs_left = self.game_state['self']

    # slice map around agent
    neighbours_x = [x+1,x-1,x,x]
    neighbours_y = [y,y,y-1,y+1]
    neighbours = [arena[neighbours_x,neighbours_y]]
    neighbours = np.array([n==0 for n in neighbours]).astype(int)[0]  
    
    # get position of nearest coin relative to agent
    coins_dist = [((xc-x)**2+(yc-y)**2)**0.5 for (xc,yc) in self.game_state['coins']]
    nearest_coin_xy = [[xc-x,yc-y] for (xc,yc) in self.game_state['coins']][np.argmin(coins_dist)]

    # get position of two nearest bombs relative to agent
    bombs = self.game_state['bombs']
    bombs = [(1,2,3),(4,5,6)]
    bombs_dist = [((xb-x)**2+(yb-y)**2)**0.5 for (xb,yb,tb) in bombs]
    nearest_bombs_xyt = [[xb-x,yb-y,tb] for (xb,yb,tb) in bombs][np.argsort(bombs_dist)[:2]]
    print(nearest_bombs_xyt)
    
    
    input_paras = np.hstack([neighbours,nearest_coin_xy])
    
    return torch.tensor([input_paras]).float()
    
def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    
    self.logger.debug(f'Set up neural net.')
    
    # imitialize neuralnetwork
    self.logger.debug(f'Load net parameters.')
    self.net = shallowQN()
    
    try:
        self.net.load_state_dict(torch.load('netparas.pt'))
        self.net.eval()
        print('Reloaded net parameters.')
    except:
        print('Could not reload net parameeters.')
        
    # learning parameters
    self.gamma = 0
    self.final_epsilon = 0.02
    self.initial_epsilon = 1
    self.epsilon = self.initial_epsilon
    self.epsilon_decay = 0.99
    self.replay_memory_size = 1000
    self.minibatch_size = 32
    
    # optimizer and loss criterion
    self.optimizer = optim.Adam(self.net.parameters(), lr=1e-1)
    self.criterion = nn.MSELoss()
    
    # initialize list for memory
    self.replay_memory = []
    self.losses = []
    self.mean_losses = []
    self.rews = []
    self.mean_rews = []    
    self.q_values = []
    self.y = []
    self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT', 'BOMB']
    
    # initial game state
    self.state = None
    self.rounds_played = 0
    
    if torch.cuda.is_available(): 
        print('Cuda is avaible.')
        self.net.cuda()
    else:
        print('Cuda is not avaible.')
        
        
def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    self.logger.info('Picking action according to rule set')

    # get game state in first round
    self.state = get_state(self)
    
    if torch.cuda.is_available(): 
        self.state = self.state.cuda()
        
    output = self.net(self.state)
    #print(output.tolist())
    
    # epsilon greedy exploration
    random_action = random.random() <= self.epsilon
    if random_action:
        self.logger.info(f"Chose random action!")
    action_index = [torch.randint(6, torch.Size([]), dtype=torch.int)
                    if random_action
                    else torch.argmax(output)][0]

    self.next_action = self.actions[action_index.tolist()]
    #self.next_action = 'BOMB'
    
    #print(self.next_action )

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    self.logger.debug(f'Encountered {self.events} game event(s)')  
    
    if self.game_state['step'] != 1:
        
        # give rewards ###############################
        
        # survived round
        reward = 1
        
        # invalid action
        if 6 in self.events:
            reward = reward - 5
            
        # bomb dropped
        if 7 in self.events:
            reward = reward - 100 
            
        # wait
        if 4 in self.events:
            reward = reward - 10

        '''                
        # move up
        if 2 in self.events:
            self.logger.debug(f'Moved Up: reward -100')
            reward = reward - 100
    
    
        # coins found
        if 11 in self.events:
            reward = reward + 100
            self.logger.info(f'Found coin!')
            
        # got killed
        if 13 in self.events:
            reward = reward - 100
        '''
        #reward = -reward
     
        self.logger.debug(f'Reward {reward}')  
    
        # fill memory ################################
        # get new state
        self.new_state = get_state(self)    
        
        # record game state, action, reward
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        action_index = torch.tensor(float(np.where([self.next_action == i for i in self.actions])[0][0]), dtype=torch.int)
        action = torch.zeros([6], dtype=torch.float32)
       
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()
            action_index = action_index.cuda()
        
        action[action_index] = 1
        action = action.unsqueeze(0)

        terminal=False
        if self.game_state['step']==s.max_steps:
            terminal = True
        self.replay_memory.append((self.state, action, reward, self.new_state,terminal))        
    
        # if replay memory is full, remove the oldest transition
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)
                
        # learning ###################################
        # sample random minibatch
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))
    
        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
 
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()       
        
        # get output for the next state
        output_1_batch = self.net(state_1_batch)
        
        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # compute Q-value
        q_value = torch.sum(self.net(state_batch) * action_batch, dim=1)

        
        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()
    
        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()
        print(q_value)
        print(y_batch)
        
        # calculate loss
        loss = self.criterion(q_value, y_batch)
        print(loss)
        
        # do backward pass
        loss.backward()
        self.optimizer.step()
        print(self.net.parameters())
    
        # set state to new state
        self.state = self.new_state
    
        
        # save parameters for evaluation
        self.y.append(np.mean(y_batch.tolist()))
        self.q_values.append(np.mean(q_value.tolist()))
        self.rews.append(reward.tolist()[0][0])
        self.mean_rews.append(np.mean(self.rews))
        self.losses.append(np.mean(loss.tolist()))
        self.mean_losses.append(np.mean(self.losses))
        
        
def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    
    # update epsilon
    self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
    
    self.rounds_played = self.rounds_played + 1
    
    if self.rounds_played%10 == 0:
        print(self.rounds_played)
    
    if self.rounds_played%100 == 0:
        torch.save(self.net.state_dict(), 'netparas.pt')
        self.logger.info(f'saved net')

    if self.rounds_played == s.n_rounds or self.rounds_played%100 == 0:  
        #after last round
        evaluation(self)


def evaluation(self):

    # saving location
    loc = "agent_code\\reduced_foofighter\\logs\\"
    
    # plot reward
    
    numrange = range(len(self.rews))
    numrange2 = range(len(self.losses))
   
    plt.figure()
    plt.plot(numrange,self.rews)
    plt.title('reward')
    plt.xlabel('transitions')
    plt.ylabel('reward')
    plt.savefig(loc+'rewards.png')  

    #plot mean reward    
    plt.figure()
    plt.title('mean reward')
    plt.plot(numrange,self.mean_rews)
    plt.xlabel('transitions')
    plt.ylabel('mean reward')
    plt.savefig(loc+'mean_rewards.png')  

    #plot qvalues
    plt.figure()
    plt.plot(numrange2,self.q_values,label='model')
    plt.plot(numrange2,self.y,label='real')
    plt.title('Q value')
    plt.xlabel('transitions')
    plt.ylabel('Q value')
    plt.legend()
    plt.savefig(loc+'q_value.png')  
            
    #plot losses
    plt.figure()
    plt.plot(numrange2,self.losses)
    plt.title('loss')
    plt.xlabel('transitions')
    plt.ylabel('loss')
    plt.savefig(loc+'losses.png')  

    #plot mean losees    
    plt.figure()
    plt.plot(numrange2,self.mean_losses)
    plt.title('mean losses')
    plt.xlabel('transitions')
    plt.ylabel('mean losses')
    plt.savefig(loc+'mean_losses.png')  

    plt.close('all')