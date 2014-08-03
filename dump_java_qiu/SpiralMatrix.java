public class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return res;
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        // maximum cycles
        int cycle = rows > cols? (cols + 1) / 2: (rows + 1) / 2;
        
        for (int count = 0; count < cycle; count++) {
            // top - move right
            for (int i = count; i < cols - count; i++) {
                res.add(matrix[count][i]);
            }
            
            // right - move down
            for (int i = count + 1; i < rows - count; i++) {
                res.add(matrix[i][cols - 1 - count]);
            }
            
            // if only one row or one column left
            if (rows - count * 2 == 1 || cols - count * 2 == 1) {
                break;
            }
            
            // bottom - move left
            for (int i = cols - 2 - count; i >= count; i--) {
                res.add(matrix[rows - 1 - count][i]);
            }
            
            // left - move up
            for (int i = rows - 2 - count; i > count; i--) {
                res.add(matrix[i][count]);
            }
        }
        return res;
    }
}
