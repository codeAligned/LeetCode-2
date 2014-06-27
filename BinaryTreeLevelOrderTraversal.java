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
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        
    	ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
    	
    	if(root == null)
    		return res;
    	
    	Queue<TreeNode> queue = new LinkedList<TreeNode>();
    	queue.offer(root);
    	int count = 1;
    	
    	while(!queue.isEmpty()){
    		int count2 = 0;
    		ArrayList<Integer> cur = new ArrayList<Integer>(); 
    		while(count >= 1) {
    			TreeNode tn = queue.peek();
    			cur.add(tn.val);
    			if(tn.left != null){
    				queue.offer(tn.left);
    				count2++;
    			}
    			if(tn.right != null){
    				queue.offer(tn.right);
    				count2++;
    			}
    			queue.poll();
    			count --;
    		}
    		res.add(cur);
    		count = count2;
    	}
    	
    	return res;
    
    }
}