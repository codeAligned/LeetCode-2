public class Solution {
    public void setZeroes(int[][] matrix) {
        
		boolean firstrowZero = false;
		boolean firstcolZero = false;
		int i = 0;
		while (i < matrix.length) {
			if (matrix[i][0] == 0) {
				firstcolZero = true;
				break;
			}
			i++;
		}
		int j = 0;
		while (j < matrix[0].length) {
			if (matrix[0][j] == 0) {
				firstrowZero = true;
				break;
			}
			j++;
		}

		for (int m = 1; m < matrix.length; m++) {
			for (int n = 1; n < matrix[0].length; n++) {
				if (matrix[m][n] == 0) {
					matrix[m][0] = 0;
					matrix[0][n] = 0;
				}
			}
		}

		for (int m = 1; m < matrix.length; m++) {
			if (matrix[m][0] == 0) {
				for (int t = 0; t < matrix[0].length; t++) {
					matrix[m][t] = 0;
				}
			}
		}
		
		for (int n = 1;n<matrix[0].length;n++){
			if(matrix[0][n]== 0){
				for (int t = 0; t < matrix.length; t++) {
					matrix[t][n] = 0;
				}	
			}
		}
		
		if(firstcolZero){
			for (int t = 0; t < matrix.length; t++) {
				matrix[t][0] = 0;
			}	
		}
		if(firstrowZero){
			for (int t = 0; t < matrix[0].length; t++) {
				matrix[0][t] = 0;
			}	
		}
	
    }
}