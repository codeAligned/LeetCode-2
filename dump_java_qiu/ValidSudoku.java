public class Solution {
    public boolean isValidSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9) {
            return false;
        }
        HashSet<Character> set = new HashSet<Character>();
        for (int i = 0; i < 9; i++) {
            set.clear();
            for (int j = 0; j < 9; j++) {
                if (!helpValidate(board[i][j], set)) {
                    return false;
                }
            }
        }
        
        for (int i = 0; i < 9; i++) {
            set.clear();
            for (int j = 0; j < 9; j++) {
                if (!helpValidate(board[j][i], set)) {
                    return false;
                }
            }
        }
        
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                set.clear();
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (!helpValidate(board[i + n * 3][j + m * 3], set)) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
    
    private boolean helpValidate(char c, HashSet<Character> set) {
        if (c == '.') {
            return true;
        }
        if (set.contains(c) || c < '0' || c > '9') {
            return false;
        }
        set.add(c);
        return true;
    }
}
