/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
  public TreeNode sortedArrayToBST(int[] num) {
        int length = num.length;
    	if(length == 0)
        	return null;
        
    	TreeNode root = build(num,0,length-1);
    	
    	return root;
    }

	private static TreeNode build(int[] num, int i, int j) {
			int mid = (j-i+1)/2 + i;
			TreeNode root = new TreeNode(num[mid]);
			if(mid > i)
				root.left = build(num, i, mid-1);
			if(mid < j)
				root.right= build(num, mid+1, j);
			return root;
	}
}