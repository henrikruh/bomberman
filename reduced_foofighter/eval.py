# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:51:08 2019

@author: Henrik
"""
import matplotlib.pylab as plt
import numpy as np
        
        
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
    plt.plot(numrange,self.mean_rews[1:])
    plt.xlabel('transitions')
    plt.ylabel('mean reward')
    plt.savefig(loc+'mean_rewards.png')  

    #plot qvalues
    plt.figure()
    plt.plot(numrange2[-1000:],self.q_values[-1000:],label='model')
    plt.plot(numrange2[-1000:],self.y[-1000:],label='real')
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
    plt.plot(numrange2,self.mean_losses[1:])
    plt.title('mean losses')
    plt.xlabel('transitions')
    plt.ylabel('mean losses')
    plt.savefig(loc+'mean_losses.png')  

    plt.close('all')