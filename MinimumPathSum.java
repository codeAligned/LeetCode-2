public class Solution {

	public int minPathSum(int[][] grid) {
		if(grid.length <= 0 || grid[0].length <= 0)
			return 0;
		int w = grid[0].length;
		int h = grid.length;
		
		for(int j = 1; j < h;j++){
			grid[j][0] += grid[j-1][0];
		}
		
		for(int i = 1;i< w;i++) {
			grid[0][i] += grid[0][i-1];
		}
		
		for(int j = 1; j < h;j++) {
			for (int i = 1; i< w;i++) {
				grid[j][i] += Math.min(grid[j][i-1], grid[j-1][i]);
			}
		}
		
		return grid[h-1][w-1];
	}

	
}