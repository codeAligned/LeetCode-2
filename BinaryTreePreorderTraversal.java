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
    
    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        ArrayList<Integer> treeList = new ArrayList<Integer>();
        preorder(treeList, root);
		return treeList;
	}
	
	public void preorder(ArrayList<Integer> tree, TreeNode root) {
		
		if(root == null)
		    return;
		
		tree.add(root.val);    
		preorder(tree, root.left);
		preorder(tree, root.right);
	}
}