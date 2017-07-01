# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 23:21:47 2017

@author: kevin
"""
import numpy as np
class RewardTuner:
    
    def __init__(self):
        return
        
    def fastWin(self, rewards):
        """Returns the tuned rewards to facilitate learning. Optimizes for quicker
        wins and slower losses."""        
        tuned = []
        for i in range(len(rewards)):
            s = np.sum(rewards[i])
            tuned.append([v/s for v in rewards[i]])
        return tuned