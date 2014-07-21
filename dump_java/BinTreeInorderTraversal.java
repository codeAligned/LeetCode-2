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
    public ArrayList<Integer> inorderTraversal(TreeNode root) {
    	ArrayList<Integer> inorder =  new ArrayList<Integer>();
    	recur(root,inorder);
    	return inorder;
    }
    
    
    private void recur(TreeNode root, ArrayList<Integer> inorder) {
    	if(root == null) {
    		return;
    	}
    	
    	recur(root.left, inorder);
    	inorder.add(root.val);
    	recur(root.right, inorder);
    	
	}
}