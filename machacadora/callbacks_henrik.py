
import numpy as np
import random
from time import time, sleep
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from settings import s
from .dqn import *


def get_state(self):
    # get arena
    arena = self.game_state['arena']

    # get agent position
    x, y, _, bombs_left, score = self.game_state['self']

    # slice map around agent
    neighbours_x = [x + 1, x - 1, x, x]
    neighbours_y = [y, y, y - 1, y + 1]
    neighbours = [arena[neighbours_x, neighbours_y]]
    neighbours = np.array([n == 0 for n in neighbours]).astype(int)[0]

    # get position of nearest coin relative to agent
    coins = [] #self.game_state['coins']

    if len(coins) == 0:
        nearest_coin_xy = [0,0]
    else:
        coins_dist = [((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
                      for (xc, yc) in coins]
        nearest_coin_xy = (np.array([[xc - x, yc - y] for (xc, yc) in coins])
                            [np.argmin(coins_dist)])

    # get position of two nearest bombs relative to agent
    bombs = self.game_state['bombs']
    while len(bombs) < 2:
        # append large numbers if no bombs are on the field
        bombs.append((-20, -20, 5))
    bombs_dist = [((xb - x) ** 2 + (yb - y) ** 2) ** 0.5 
                  for (xb, yb, tb) in bombs]
    nearest_bombs_xyt = np.hstack(np.array([[xb - x, yb - y, tb] 
                        for (xb, yb, tb) in bombs])
                        [np.argsort(bombs_dist)[:2]])
    
    # get positions of nearest enemy
    others = self.game_state['others']
    if len(others) == 0:
        # append large numbers if no others are on the field
        others.append((-30, -30, 0, 0, 0))
        
    others_dist = [((xo - x) ** 2 + (yo - y) ** 2) ** 0.5 
                   for (xo, yo, n, b, sc) in others]

    nearest_other_xy = (np.array([[xo - x, yo - y] 
                        for (xo, yo, n, b, sc) in others])
                        [np.argmin(others_dist)])

    input_paras = np.hstack([neighbours, nearest_coin_xy,
                            nearest_bombs_xyt, nearest_other_xy])

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

    # initialize neural network
    self.logger.debug(f'Load net parameters.')
    self.net = onelayer_14input()
    self.net_name = self.net.name
    try:
        self.net.load_state_dict(torch.load(os.path.join(self.net_name+'.pt')))
        #self.net.eval()
        print('Reloaded net parameters.')
    except:
        print('Could not reload net parameters.')

    if torch.cuda.is_available():
        print('Cuda is avaible.')
        self.net.cuda()
    else:
        print('Cuda is not avaible.')


    # logging ################################
    basic_path = 'Bomberman_logger\\'
    counter = 'lr1e-4_modinput_slowedecay\\'
    path = os.path.join(basic_path+self.net_name, counter)

    self.writer = SummaryWriter(os.path.join(path, 'loss'))
    self.writer2 = SummaryWriter(os.path.join(path, 'q-value'))
    self.writer3 = SummaryWriter(os.path.join(path, 'reward'))
    self.writer4 = SummaryWriter(os.path.join(path, 'epsilon'))
    self.writer5 = SummaryWriter(os.path.join(path, 'total_reward'))

    # learning parameters ##########################
    self.gamma = 0
    self.final_epsilon = 0.02
    self.initial_epsilon = 1
    self.epsilon = self.initial_epsilon
    self.epsilon_decay = 0.9995
    self.replay_memory_size = 1000
    self.minibatch_size = 32

    # optimizer and loss criterion
    self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
    self.criterion = nn.MSELoss()

    # initialize list for memory
    self.replay_memory = []
    self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

    # initial game state
    self.state_agent = None
    self.rounds_played = 0
    self.totaltransitions = 0
    self.loss_list = []
    self.q_list = []
    self.reward_list = []
    self.total_reward = 0


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

    # epsilon greedy exploration
    random_action = random.random() <= self.epsilon
    if random_action:
        self.logger.info(f"Performed random action!")
    action_index = [torch.randint(6, torch.Size([]), dtype=torch.int)
                    if random_action
                    else torch.argmax(output)][0]

    self.next_action = self.actions[action_index.tolist()]


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    self.totaltransitions += 1

    # give rewards ###############################
    # survived round
    reward = 1

    # wait
    if 4 in self.events:
        reward = reward - 10

    # places bomb
    if 7 in self.events:
        reward = reward - 100

    # invalid action
    if 6 in self.events:
        reward = reward - 5

    '''
    # coins found
    if 11 in self.events:
        reward = reward + 20
        self.logger.info(f'Found coin!')

    # kills opponent
    if 12 in self.events:
        reward = reward + 200

    # got killed
    if 13 in self.events:
        reward = reward - 10
    '''
    self.total_reward += reward

    # fill memory ################################
    # get new state
    self.new_state = get_state(self)

    # record game state, action, reward
    reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
    action_index = torch.tensor(float(np.where([self.next_action == i for i in self.actions])[0][0]), dtype=torch.int)
    action = torch.zeros([6], dtype=torch.float32)

    if torch.cuda.is_available():  # put on GPU if CUDA is available
        action = action.cuda()
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        action_index = action_index.cuda()

    action[action_index] = 1
    action = action.unsqueeze(0)

    terminal = False
    if self.game_state['step'] == s.max_steps:
        terminal = True
    self.replay_memory.append((self.state, action, reward, self.new_state, terminal))

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

    # get output for the next state
    output_1_batch = self.net(state_1_batch)

    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                              else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                              for i in range(len(minibatch))))

    # extract Q-value
    q_value = torch.sum(self.net(state_batch) * action_batch, dim=1)

    # PyTorch accumulates gradients by default, so they need to be reset in each pass
    self.optimizer.zero_grad()

    # returns a new Tensor, detached from the current graph, the result will never require gradient
    y_batch = y_batch.detach()

    # calculate loss
    loss = self.criterion(q_value, y_batch)

    # do backward pass
    loss.backward()
    self.optimizer.step()

    # set state to new state
    self.state = self.new_state

    # output ###############################
    self.loss_list.append(loss.item())
    self.q_list.append(q_value.mean().item())
    self.reward_list.append(reward.item())

    if self.totaltransitions%100==0:
        loss_mean = np.mean(np.asarray(self.loss_list))
        q_mean = np.mean(np.asarray(self.q_list))
        reward_mean = np.mean(np.asarray(self.reward_list))
        print('After {} batches, mean loss: {}'.format(self.totaltransitions, loss_mean))
        self.writer.add_scalar('loss per batch', loss_mean, self.totaltransitions)
        self.writer2.add_scalar('q-value per batch', q_mean, self.totaltransitions)
        self.writer3.add_scalar('reward per batch', reward_mean, self.totaltransitions)
        self.loss_list, self.q_list, self.reward_list = [], [], []
        self.writer4.add_scalar('epsilon', self.epsilon, self.totaltransitions)


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    reward_update(self)

    self.rounds_played = self.rounds_played + 1

    self.writer5.add_scalar('total_reward_per_round', self.total_reward, self.rounds_played)
    self.total_reward = 0

    # update epsilon
    self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    if self.rounds_played % 100 == 0:
        torch.save(self.net.state_dict(), os.path.join(self.net_name+'.pt'))
        self.logger.info(f'saved net')

    if self.rounds_played == s.n_rounds:
        torch.save(self.net.state_dict(), os.path.join(self.net_name+'.pt'))
        self.logger.info(f'saved net')
