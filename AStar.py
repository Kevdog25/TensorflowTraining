# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 08:42:56 2017

@author: kevin
"""
import queue
import numpy as np

def getHash(array):
    return ''.join(str(a) for a in array)
    
class Node:
    def __init__(self, state, parent = None, action = None):
        self.State = state
        self.StateHash = getHash(state)
        self.Parent = parent
        self.Action = action
        return
        
    def __eq__(self, other):
        if other is None: return False
        return self.StateHash == other.StateHash
        
    def __hash__(self):
        return self.StateHash
    
    def __lt__(self, other):
        return self.StateHash < other.StateHash
    def __gt__(self, other):
        return self.StateHash > other.StateHash
    def __str__(self):
        
        return str(self.State) + '|' + str(self.Action) + '|' + str(self.Parent == None)
        
class AStar:
    
    @staticmethod
    def getAction(node):
        act = None
        while node.Parent is not None:
            act = node.Action
            node = node.Parent
        return act
        
    def __init__(self, evaluator, maxDepth, timeLimit):
        self.Evaluator = evaluator
        self.MaxDepth = maxDepth
        self.TimeLimit = timeLimit
        self.Root = None
        return        
        
    def findMove(self, state, game):
        self.Root = Node(state)
        self.Explored = {getHash(state)}
        self.Q = queue.PriorityQueue()
        self.expandNode(self.Root, game)
        bestMove = (-1000000000, None)
        n = 0
        while not self.Q.empty() and n < self.MaxDepth:
            nodeScore = self.Q.get(False)
            value = -nodeScore[0]
            if value > bestMove[0]:
                bestMove = nodeScore
            elif value == bestMove[0] and np.random.rand() < 0.5:
                bestMove = nodeScore
            if int(value) == 1:
                break
            if int(value) == -1:
                continue
            n += 1
            self.expandNode(nodeScore[1], game)
        return self.getAction(bestMove[1])
        
    
    def expandNode(self, node, game):
        actions = game.getValidActions(node.State)
        for i in [j for j in range(actions.shape[1]) if actions[0,j] > 0]:
            action = np.zeros(actions.shape)
            action[0, i] = 1
            state, end = game.getNextState(node.State, action)
            h = getHash(state)
            if not h in self.Explored:
                self.Explored.add(h)
                self.Q.put((-self.Evaluator(state, end), Node(state, node, action)),False)
                

            
        return