# Sudoku Solver (performance test included)
import time
import copy

class Solution1:
    def modifyMask(self, rmask,cmask,bmask,i,j,b,change):
        rmask[i] ^= change
        cmask[j] ^= change
        bmask[b] ^= change

    def dfs(self, board, k, rmask, cmask, bmask):
        if k == 81:
            return True
        i,j=k/9,k%9
        b = i / 3 * 3 + j / 3
        if board[i][j] != '.':
            return self.dfs(board,k+1,rmask,cmask,bmask)
        for digit in range(9):
            change = 1 << digit
            if rmask[i] & change == 0 and cmask[j] & change == 0 and bmask[b] & change == 0:
                self.modifyMask(rmask,cmask,bmask,i,j,b,change)
                board[i][j] = str(digit+1)
                if self.dfs(board,k+1,rmask,cmask,bmask):
                    return True
                board[i][j] = '.'
                self.modifyMask(rmask,cmask,bmask,i,j,b,change)
        return False

    def solveSudoku(self, board):
        startT = time.clock()
        rmask,cmask,bmask=[0]*9,[0]*9,[0]*9
        for i in range(9):
            for j in range(9):
                b = i / 3 * 3 + j / 3
                if board[i][j] != '.':
                    change = 1 << (int(board[i][j]) - 1)
                    self.modifyMask(rmask,cmask,bmask,i,j,b,change)
        self.dfs(board,0,rmask,cmask,bmask)
        print "Total time 1: ", time.clock() - startT

class Solution2:
    # @param board, a 9x9 2D array
    # Solve the Sudoku by modifying the input board in-place.
    # Do not return any value.
    def solveSudoku(self, board):
        startT = time.clock()
        self.dfs(board, 0)
        print "Total time 2: ", time.clock() - startT


    def dfs(self, board, pos):
        n = len(board)
        if pos == n * n: return True
        x = pos // n; y = pos % n
        if board[x][y] == '.':
            for k in "123456789":
                board[x][y] = k
                if self.isValid(board, x, y) and self.dfs(board, pos + 1):
                    return True
                board[x][y] = '.'
        else:
            if self.dfs(board, pos + 1): return True
        return False

    def isValid(self, board, x, y):
        tmp = board[x][y]
        for i in xrange(9):
            if i != x and board[i][y] == tmp:
                return False
            if i != y and board[x][i] == tmp:
                return False

        xx = x // 3 * 3; yy = y // 3 *3
        for i in xrange(xx, xx + 3):
            for j in xrange(yy, yy + 3):
                if i != x and j != y and board[i][j] == tmp:
                    return False
        return True


sudoku = [".....7...",".4..812..","...9...1.","..53...72","293....5.",".....53..","8...23...","7...5..4.","531.7...."]
board = [[] for i in xrange(9)]
for row in xrange(len(sudoku)):
    for i in xrange(len(sudoku[0])):
        board[row].append(sudoku[row][i])

def test1():
    board1 = copy.deepcopy(board)
    Solution1().solveSudoku(board1)

def test2():
    board2 = copy.deepcopy(board)
    Solution2().solveSudoku(board2)

from timeit import Timer
t1 = Timer("test1()","from __main__ import test1")
t2 = Timer("test2()","from __main__ import test2")
# print t1.timeit(20)
# print t2.timeit(20)

print t1.repeat(1, 10)
print t2.repeat(1, 10)

#
# 28. return the number of different ways to climb stairs, each level can move
# 4 steps at most.


def climbStairs(num):
    if num <= 4:
        resArray = [0 for i in xrange(5)]
        initArray(resArray)
        return resArray[num]
    elif num >= 5:
        resArray = [0 for i in xrange(num + 1)]
        initArray(resArray)
        for i in xrange(5, num + 1):
            resArray[i] = resArray[i - 1] + resArray[i - 2] + \
                resArray[i - 3] + resArray[i - 4]
        return resArray[num]


def initArray(resArray):
    for i in xrange(len(resArray)):
        resArray[0] = 1
        resArray[1] = 1
        resArray[2] = 2
        resArray[3] = resArray[2] + resArray[1] + resArray[0]
        resArray[4] = resArray[3] + resArray[2] + resArray[1] + resArray[0]
    return resArray

#
# 27. Interview question 1
# change the nth digit of binary value of an integer to 1


def solution(a, offset):
    tmp = a >> (offset - 1)
    if tmp % 2 == 1:
        return a
    elif tmp % 2 == 0:
        xorVal = 1 << (offset - 1)
        return a ^ xorVal
