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
    public boolean isSymmetric(TreeNode root) {
		if (root == null)
		    return true;
		if (root.left == null && root.right==null) 
			return true;
		if (root.left == null || root.right == null)
			return false;
		
		return check(root.left, root.right);
	}

	private boolean check(TreeNode Left, TreeNode Right) {
		if (Left == null && Right ==null) 
			return true;
		if (Left == null || Right == null)
			return false;
		
		return Left.val == Right.val && check(Left.left, Right.right) && check(Left.right, Right.left);
	}
}