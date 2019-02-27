
import numpy as np
import random
from time import time, sleep
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from settings import s
from .dqn import DQN


def get_state(self):
    # Gather information about the game state
    arena = self.game_state['arena'][1:s.cols - 1, 1:s.rows - 1]

    #print('arena')
    #print(arena)

    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x, y) for (x, y, t) in bombs]
    bombs = self.game_state['bombs']
    others = [(x, y) for (x, y, n, b, sc) in self.game_state['others']]
    others_x = [x for (x, y, n, b, sc) in self.game_state['others']]
    others_y = [y for (x, y, n, b, sc) in self.game_state['others']]
    coins = self.game_state['coins']
    coins_x = [x for (x, y) in self.game_state['coins']]
    coins_y = [y for (x, y) in self.game_state['coins']]
    bomb_map = np.ones([s.cols, s.rows]) * 5
    for xb, yb, t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    bomb_map = bomb_map[1:s.cols - 1, 1:s.rows - 1]

    # print('bombs')
    # print(bomb_map)

    agents_map = np.zeros([s.cols, s.rows])
    agents_map[x, y] = 1
    agents_map[others_x, others_y] = -1
    agents_map = agents_map[1:s.cols - 1, 1:s.rows - 1]

    # print('agents')
    # print(agents_map)

    coins_map = np.zeros([s.cols, s.rows])
    coins_map[coins_x, coins_y] = 1
    coins_map = coins_map[1:s.cols - 1, 1:s.rows - 1]

    # print('coins')
    # print(coins_map)

    joint_map = self.game_state['arena']
    joint_map[x, y] = 2
    joint_map[others_x, others_y] = 3
    joint_map[coins_x, coins_y] = 4
    joint_map = joint_map[1:s.cols - 1, 1:s.rows - 1]

    # print('joint_map')
    # print(joint_map)

    # concatenate state

    return torch.tensor([[arena, bomb_map, agents_map, coins_map]]).float()


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).

    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    """

    self.logger.debug(f'Set up neural net.')

    # imitialize neuralnetwor
    self.logger.debug(f'Load net parameters.')
    self.net = DQN()
    self.parameters = self.net.parameters()
    #print(self.parameters)
    self.net.load_state_dict(torch.load('netparas.pt'))
    self.net.eval()

    # learning parameters
    self.gamma = 0.99
    self.final_epsilon = 0.0001
    self.initial_epsilon = 0.2
    self.epsilon = self.initial_epsilon
    self.replay_memory_size = 10000
    self.minibatch_size = 32

    # optimizer and loss criterion
    self.optimizer = optim.Adam(self.net.parameters(), lr=1e-6)
    self.criterion = nn.MSELoss()

    # initialize list for memory
    self.replay_memory = []
    self.actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

    # initial game state
    self.state = None
    self.rounds_played = 0


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

    self.logger.info('Picking action according to rule set')

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))

    # Check which moves make sense at all
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x,y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for xb,yb,t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            self.next_action = a
            break

    # Keep track of chosen action for cycle detection
    if self.next_action == 'BOMB':
        self.bomb_history.append((x,y))
    """

    # get game state in first round
    self.state = get_state(self)

    output = self.net(self.state)
    # print(output.tolist())
    # epsilon greedy exploration
    random_action = random.random() <= self.epsilon
    if random_action:
        self.logger.info(f"Performed random action!")
    action_index = [torch.randint(6, torch.Size([]), dtype=torch.int)
                    if random_action
                    else torch.argmax(output)][0]

    self.next_action = self.actions[action_index.tolist()]

    # self.next_action = 'BOMB'
    # print(self.next_action )


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.

    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    # give rewards ###############################

    # survived round
    reward = -1

    # places bomb

    # invalid action
    if 6 in self.events:
        reward = reward - 200

    # coins found
    if 11 in self.events:
        reward = reward + 20
        self.logger.info(f'Found coin!')

    # kills opponent
    if 12 in self.events:
        reward = reward + 200

    # got killed
    if 13 in self.events:
        reward = reward - 500


    # fill memory ################################
    # get new state
    self.new_state = get_state(self)

    # record game state, action, reward
    reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
    action = torch.tensor([[float(np.where([self.next_action == i for i in self.actions])[0][0])]])
    terminal = False
    if self.events in [14, 16]:
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

    # perform next action
    # act(self)



def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    reward_update(self)

    self.rounds_played = self.rounds_played + 1

    if self.rounds_played == s.n_rounds:
        torch.save(self.net.state_dict(), 'netparas.pt')
        self.logger.info(f'saved net')