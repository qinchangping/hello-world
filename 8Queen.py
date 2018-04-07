__author__ = 'qcp'
# -*- coding:utf-8 -*-

import numpy as np


class Board(object):
	def __init__(self, size):
		self.board_size = size
		self.board = np.zeros((self.board_size, self.board_size))
		self.queens = []
		self.num_ans = 0
		self.TOTAL = 8 ** 8


	def get_queens(self):
		for i in range(self.TOTAL):
			self.queens = []
			self.queens.append(i % 8)
			total = i // 8
			for _ in range(7):
				yu = total % 8
				self.queens.append(yu)
				total = total // 8
			# self.queens.append(i % 8)
			if self.is_legal():
				print('the queens places are: ', self.queens)
				self.draw_board()
				self.num_ans += 1

	def is_legal(self):
		flag = True
		qu = self.queens[:]
		for i in range(8):
			q = qu[i]
			for j in range(8):
				if j != i:
					if q == qu[j] or q + i == qu[j] + j or q - i == qu[j] - j:
						flag = False
		return flag


	def draw_board(self):
		self.board = np.zeros((self.board_size, self.board_size))
		line = 0
		for queen in self.queens:
			self.board[line, queen] = 1
			line += 1
		print('The %d th board is:' % self.num_ans)
		print(self.board)


def run1():
	'''暴力穷举法求解'''
	a = Board(8)
	a.get_queens()
	# a.search()
	print('The total number of answers is: ', a.num_ans)


class Queen(object):
	def __init__(self, size):
		self.board_size = size
		self.board = np.zeros((self.board_size, self.board_size))
		# self.queens = [[x, 0] for x in range(8)]
		self.queens = [[0, 0]] * 8
		self.num_queens = 0
		self.num_ans = 0

	# self.TOTAL = 8 ** 8

	def draw_board(self):
		self.board = np.zeros((self.board_size, self.board_size))
		for queen in self.queens:
			self.board[queen[0], queen[1]] = 1
		print('The %d th board is:' % self.num_ans)
		print(self.board)

	def is_next_queen(self, q):
		flag = True
		if q[0] > 0:
			queens = self.queens[:q[0]]  # 上n行的皇后
			for queen in queens:
				if q[0] == queen[0] or q[1] == queen[1] or q[0] + q[1] == queen[0] + queen[1] or \
										q[0] - q[1] == queen[0] - queen[1]:
					flag = False
		return flag

	def set_queen(self, row, col):
		if row == 0:
			self.queens[0] = [row, col]
			return True
		elif self.is_next_queen([row, col]):
			self.queens[row] = [row, col]
			return True
		else:
			return False


	def get_queens(self, row):
		if row == 8:
			print('the queens places are: ', self.queens)
			self.draw_board()
			self.num_ans += 1
			return

		for i in range(8):
			if self.set_queen(row, i):
				# self.queens[row] = [row, i]
				self.get_queens(row + 1)
			self.queens[row] = [row, 0]


def run2():
	'''迭代法'''
	a = Queen(8)
	a.get_queens(0)
	print('The total number of answers is: ', a.num_ans)


run2()
