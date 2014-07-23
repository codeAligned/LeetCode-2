public class Solution {
    public void solveSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9) {
            return;
        }
        helpSolveSudoku(board);
    }
    
    private boolean helpSolveSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    continue;
                }
                for (int num = 0; num < 9; num++) {
                    board[i][j] = (char)(num + '1');
                    if (isValidSudoku(i, j, board) && helpSolveSudoku(board)) {
                        return true;
                    }
                    board[i][j] = '.';
                }
                return false;
            }
        }
        return true;
    }
    
    private boolean isValidSudoku(int row, int col, char[][] board) {
        for (int i = 0; i < 9; i++) {
            if (i != col && board[row][col] == board[row][i]) {
                return false;
            }
        }
        for (int i = 0; i < 9; i++) {
            if (i != row && board[row][col] == board[i][col]) {
                return false;
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int x = row / 3 * 3 + i;
                int y = col / 3 * 3 + j;
                if ( x != row && y != col && 
                    board[row][col] == board[x][y]) {
                    return false;
                }
            }
        }
        return true;
    }
}
