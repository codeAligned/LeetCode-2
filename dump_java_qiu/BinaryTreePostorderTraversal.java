public class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) {
            return result;
        }
        
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.push(root);
        TreeNode prev = null;
        
        while (!stack.isEmpty()) {
            TreeNode curt = stack.peek();
            // traversing down the tree
            if (prev == null || curt == prev.left || curt == prev.right) {
                if (curt.left != null) {
                    stack.push(curt.left);
                } else if (curt.right != null) {
                    stack.push(curt.right);
                }
            // traversing up the tree from the left
            } else if (prev == curt.left) {
                if (curt.right != null) {
                    stack.push(curt.right);
                }
            // traversing up the tree from the right   
            } else {
                result.add(curt.val);
                stack.pop();
            }
            prev = curt;
        }
        return result;
    }
}
