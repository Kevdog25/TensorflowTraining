# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 20:58:00 2017

@author: kevin
"""

import numpy as np
from b02f4d5609550a0a04878dc5a54f9c2b import game
from AStar import *

def eval(s, e):
    return e
    if int(e) != 0:
        return e
    return np.mean(s)

nWins = 0
nGames = 10
maxMoves = 100

player = AStar(eval, 50, 100)

new_game = game()
print('starting')
for i in range(nGames):
    next_board = new_game.newGame()
    print('New Game')
    for j in range(maxMoves):
        print(next_board)
        action = player.findMove(next_board, new_game)
        next_board, e = new_game.getNextState(next_board,action)
        if int(e) != 0:
            break
    if e > 0:
        print('Won!')
        nWins += 1
    else:
        print('Lose!')
    print(nWins/(i+1))
        