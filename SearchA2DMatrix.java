public class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        
		int h = matrix.length;
		int w = matrix[0].length;
		if (h <= 0 || w <= 0)
			return false;

		int start = 0;
		int end = h * w-1;
		int mid;

		while (start <= end) {
			mid = (end + start) / 2;
			int mid_X = mid / w;
			int mid_Y = mid % w;
			if (matrix[mid_X][mid_Y] == target)
				return true;
			if (matrix[mid_X][mid_Y] < target) {
				start = mid + 1;
			} else {
				end = mid - 1;
			}
		}
		return false;
	
    }
}