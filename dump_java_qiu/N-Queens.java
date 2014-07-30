public class Solution {
    public List<String[]> solveNQueens(int n) {
        List<String[]> result = new ArrayList<String[]>();
        if (n <= 0) {
            return result;
        }
        
        List<Integer> sol = new ArrayList<Integer>();
        helpSolveQueens(n, sol, result);
        return result;
    }
    
    private void helpSolveQueens(int n, List<Integer> sol, List<String[]> result) {
        int row = sol.size();
        
        if (row == n) {
            result.add(drawBoard(sol));
            return;
        }
        
        for (int i = 0; i < n; i++) {
            if (isValid(i, sol)) {
                sol.add(i);
                helpSolveQueens(n, sol, result);
                sol.remove(sol.size() - 1);
            }
        }
    }
    
    private boolean isValid(int col, List<Integer> sol) {
        // check col & intersection
        int row = sol.size();

		for (int i = 0; i < row; i++) {
			if (sol.get(i) == col) {
				return false;
			}

			if (row - i == Math.abs(sol.get(i) - col)) {
				return false;
			}
        }
        
        return true;
    }
    
    private String[] drawBoard(List<Integer> sol) {
        String[] res = new String[sol.size()];
        for (int i = 0; i < sol.size(); i++) {
            res[i] = "";
            for (int j = 0; j < sol.size(); j++) {
                if (j != sol.get(i)) {
                    res[i] += '.';
                } else {
                    res[i] += 'Q';
                }
            }
        }
        return res;
    }
}
