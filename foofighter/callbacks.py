
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s

import torch
import torch.nn as nn
import torch.nn.functional as F

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

class Net(nn.Module):

    # initialize neural network
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #(w-4)/2
        x = F.relu(self.conv2(x)) #(w-4)/2
        x = x.view(-1, 16 * 7 * 7 ) #(16*w*h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    
    self.net = Net()
      
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    
            
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

    # Gather information about the game state
    arena = self.game_state['arena'][1:s.cols-1,1:s.rows-1] 
    print('arena')
    print(arena)
    x, y, _, bombs_left = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    bombs = self.game_state['bombs']
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    others_x = [x for (x,y,n,b) in self.game_state['others']]
    others_y = [y for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']
    coins_x = [x for (x,y) in self.game_state['coins']]
    coins_y = [y for (x,y) in self.game_state['coins']]
    bomb_map = np.ones([s.cols,s.rows]) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    bomb_map = bomb_map[1:s.cols-1,1:s.rows-1] 
    
    print('bombs')
    print(bomb_map)

    agents_map = np.zeros([s.cols,s.rows])
    agents_map[x,y]=1
    agents_map[others_x,others_y]=-1
    agents_map = agents_map[1:s.cols-1,1:s.rows-1] 
    
    print('agents')
    print(agents_map)

    coins_map = np.zeros([s.cols,s.rows])
    coins_map[coins_x,coins_y]=1
    coins_map = coins_map[1:s.cols-1,1:s.rows-1]     
    
    print('coins')
    print(coins_map)
    
    
    joint_map = self.game_state['arena']
    joint_map[x,y]=2
    joint_map[others_x,others_y]=3
    joint_map[coins_x,coins_y]=4
    joint_map = joint_map[1:s.cols-1,1:s.rows-1]     
    
    print('joint_map')
    print(joint_map)
    
    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT','BOMB','WAIT']
    output = self.net(torch.tensor([[arena,bomb_map,agents_map,coins_map]]).float())

    
    print(output.tolist())
    self.next_action = actions[output.argmax().tolist()]
    print(self.next_action )

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
