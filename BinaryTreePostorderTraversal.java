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
    public ArrayList<Integer> postorderTraversal(TreeNode root) {
		
		Stack<TreeNode> stack1 = new Stack<TreeNode>();
		Stack<TreeNode> stack2 = new Stack<TreeNode>();
		ArrayList<Integer> res = new ArrayList<Integer>();
		
		if(root == null)
		    return res;
		    
		stack1.push(root);
		
		while(!stack1.isEmpty()){
			TreeNode top = stack1.pop();
			if(top.left != null){
				stack1.push(top.left);
			}
			if(top.right != null){
				stack1.push(top.right);
			}
			
			stack2.push(top);
		}
		
		while(!stack2.isEmpty()){
			res.add(stack2.pop().val);
		}
		
		return res;
	
    }
}