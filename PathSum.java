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
    public boolean hasPathSum(TreeNode root, int sum) {
        
		if (root == null)
			return false;

		Stack<Integer> accu = new Stack<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		stack.push(root);
		accu.push(root.val);

		while (!stack.isEmpty()) {
			TreeNode tn = stack.pop();
			int acc = accu.pop();
			
			if(tn.left == null && tn.right == null && acc == sum)
				return true;
			
			if (tn.right != null) {
				stack.push(tn.right);
				accu.push(acc+tn.right.val);
			}
			if (tn.left != null) {
				stack.push(tn.left);
				accu.push(acc+tn.left.val);
			}
		}
		return false;
	
    }
}