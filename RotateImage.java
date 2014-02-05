public class Solution {
    public void rotate(int[][] matrix) {
        
    	int n = matrix.length;
        
    	int p = 0;
        int q = n-1;
        int tmp;
        
        while(p<q){
        	for(int i = 0; i<q-p;i++){
        		tmp = matrix[p][p+i];
        		matrix[p][p+i] = matrix[q-i][p];
        		matrix[q-i][p] = matrix[q][q-i];
        		matrix[q][q-i] = matrix[p+i][q];
        		matrix[p+i][q] = tmp;
        	}
        	p++;
        	q--;
        }
    
    }
}