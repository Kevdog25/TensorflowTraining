import numpy as np

class player():
	def __init__(self, position=1):
		self.position = position
		self.board = self.update_board(self.position)
		self.win_state = 0	

	def move(self, movement):
		input = 0
		if movement[0,0] == 1: 
			input = -1
		elif movement[0,1] == 1:
			input = 1

		if input == 1 or input == -1:
			self.position += input
			if self.position not in range(0,3):
				self.change_win(abs(self.position)/self.position)
			else:
				self.board = self.update_board(self.position)

	def change_win(self, new_win_state):
		self.win_state = new_win_state

	def get_win_state(self):
		return self.win_state

	def update_board(self, pos):
		board_state = np.zeros((1,3))
		board_state[0,pos] = 1
		return board_state

	def get_board(self):
		return self.board

	def reset(self):
		self.__init__()

class game():
	def __init__(self):
		self.current_player = player()
		self.history = [self.current_player.get_board()]

	def getPossibleMoves(self):
		return 2

	def getCurrentState(self):
		return self.current_player.get_board()

	def getNextState(self, board, movement):
		faux_player = player(np.nonzero(board[0,:])[0][0])
		faux_player.move(movement)
		return faux_player.get_board(), faux_player.get_win_state()

	def movePlayer(self, movement):
		self.current_player.move(movement)
		self.history.append(self.current_player.get_board())

	def resetGame(self):
		self.current_player.reset()
		self.history = [self.current_player.get_board()]
