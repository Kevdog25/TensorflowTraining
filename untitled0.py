# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 20:58:00 2017

@author: kevin
"""

import numpy as np
from b02f4d5609550a0a04878dc5a54f9c2b import game

new_game = game()
next_board = new_game.newGame()
print('Current Board:\n',next_board)

while True:
        move = str(input('make a move (L or R):'))
        a = np.zeros((1,2))
        if move.upper() == 'L':
                a[0,0] = 1
        elif move.upper() == 'R':
                a[0,1] = 1

        next_board, win = new_game.getNextState(next_board, a)
        print('\nWin Status:', win, '\nCurrent Board:\n',next_board)

        if win != 0:
                next_board = new_game.newGame()
                win = 0
                print('NEW GAME!')