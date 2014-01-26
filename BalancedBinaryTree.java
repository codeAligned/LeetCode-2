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

	
	public boolean isBalanced(TreeNode root) {
		if(root == null)
			return true;
		else if(root.left == null && root.right == null)
			return true;
		
    	return isBalanced(root.left) && isBalanced(root.right)
				&& Math.abs((getHeight(root.left) - getHeight(root.right))) <= 1;
	}

	private int getHeight(TreeNode root) {
		if(root == null)
			return 0;
		else if (root.left == null && root.right == null)
			return 1;
		else 
			return Math.max(getHeight(root.left), getHeight(root.right))+1;
			
	}

}