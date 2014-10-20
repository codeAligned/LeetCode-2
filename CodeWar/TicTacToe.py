__author__ = 'leon'


def isSolved(board):
    win_X = False
    win_O = False

    unfinished = False
    for i in xrange(3):
        for j in xrange(3):
            if board[i][j] == 0:
                unfinished = True

    for i in xrange(3):
        if board[i][0] == board[i][1] and board[i][1] == board[i][2]:
            if board[i][0] == 1:
                win_X = True
            elif board[i][0] == 2:
                win_O = True

    for j in xrange(3):
        if board[0][j] == board[1][j] and board[1][j] == board[2][j]:
            if board[0][j] == 1:
                win_X = True
            elif board[0][j] == 2:
                win_O = True

    if (board[0][0] == board[1][1] and board[1][1] == board[2][2]) or (
                    board[2][0] == board[1][1] and board[1][1] == board[0][2]):
        if board[1][1] == 1:
            win_X = True
        elif board[1][1] == 2:
            win_O = True

    if win_O and win_X:
        return 0
    elif (not win_O) and (not win_X):
        if unfinished:
            return -1
        else:
            return 0
    elif win_X:
        return 1
    elif win_O:
        return 2


print isSolved([[0, 0, 1], [0, 1, 2], [2, 1, 0]])