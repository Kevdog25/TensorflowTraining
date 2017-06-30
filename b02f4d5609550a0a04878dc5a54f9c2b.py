import numpy as np

class player():
	def __init__(self, position=5, game_state=np.zeros((1,2))):
		self.position = position
		self.state = self.update_state(position, game_state)
		self.win_state = 0	

	def move(self, movement):
		input = 0
		if movement[0,0] == 1: 
			input = -1
		elif movement[0,1] == 1:
			input = 1

		if input == 1 or input == -1:
			self.position += input
			if self.position == 8:
				self.state[0,11] = 1
			
			if self.position == 3 and self.state[0,11]:
				self.state[0,12] = 1

			if self.position not in range(0,11):
				if self.state[0,12]:
					self.change_win(1)
				else:
					self.change_win(-1)
			else:
				self.state = self.update_state(self.position, self.state[:,11:13])

	def change_win(self, new_win_state):
		self.win_state = new_win_state

	def get_win_state(self):
		return self.win_state

	def update_state(self, pos, conditions):
		game_state = np.concatenate((np.zeros((1,11)), conditions), axis=1)
		game_state[0,pos] = 1
		return game_state

	def get_state(self):
		return self.state

	def reset(self):
		self.__init__()

class game():
	def getPossibleMoves(self):
		return 2

	def getNextState(self, state, movement):
		board = np.array([state[0,:-2]])
		game_variables = np.array([state[0,-2:]])
		
		faux_player = player(np.nonzero(board[0])[0][0], game_state=game_variables)
		faux_player.move(movement)
		next_state = faux_player.get_state()
		return next_state, faux_player.get_win_state()
		
	def newGame(self):
		new_player = player()
		return new_player.state
