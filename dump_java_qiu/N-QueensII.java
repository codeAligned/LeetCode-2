public class Solution {
    private int solutions = 0;
    
    public int totalNQueens(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] columns = new int[n];
        helpSolveNQueens(columns, 0, n);
        return solutions;
    }
    
    private void helpSolveNQueens(int[] columns, int row, int n) {
        if (row == n) {
            solutions++;
            return;
        }
        
        for (int i = 0; i < n; i++) {
            if (isValid(columns, row, i)) {
                columns[row] = i;
                helpSolveNQueens(columns, row + 1, n);
                columns[row] = 0;
            }
        }
    }
    
    private boolean isValid(int[] columns, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (columns[i] == col) {
                return false;
            }
        }
        
        for (int i = 0; i < row; i++) {
            if (Math.abs(row - i) == Math.abs(col - columns[i])) {
                return false;
            }
        }
        return true;
    }
}
