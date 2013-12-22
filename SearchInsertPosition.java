/**
 * Search Insert Position Total Accepted: 4930 Total Submissions: 14401 My Submissions
 * Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

 * You may assume no duplicates in the array.

 * Here are few examples.
 * [1,3,5,6], 5 → 2
 * [1,3,5,6], 2 → 1
 * [1,3,5,6], 7 → 4
 * [1,3,5,6], 0 → 0

 */
 
public class Solution {
    public int searchInsert(int[] A, int target) {

		int index = 0;		
		Stack<Integer> s = new Stack<Integer>();
		
		for (int i = 0; i < A.length; i++) {
			if (A[i] < target) {
				s.push(A[i]);
				index++;
			} 

		}
		return index;

	}
}