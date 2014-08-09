// 1. Single number

public class Solution {
    public int singleNumber(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }
        int result = A[0];
        for (int i = 1; i < A.length; i++) {
            result ^= A[i];
        }
        return result;
    }
}

// 2. Maximum Depth of Binary Tree
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
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }
}

// 3. Binary Tree Preorder Traversal
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
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) {
            return result;
        }
        
        Stack<TreeNode> nodes = new Stack<TreeNode>();
        nodes.push(root);
        
        while (!nodes.isEmpty()) {
            TreeNode curt = nodes.pop();
            result.add(curt.val);
            
            if (curt.right != null) {
                nodes.push(curt.right);
            }
            if (curt.left != null) {
                nodes.push(curt.left);
            }
        }
        return result;
    }
}

// 4. Binary Tree Inorder Traversal
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
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> nodeStack = new Stack<TreeNode>();
        TreeNode curt = root;
        
        while (!nodeStack.isEmpty() || curt != null) {
            if (curt != null) {
                nodeStack.add(curt);
                curt = curt.left;
            } else {
                curt = nodeStack.pop();
                result.add(curt.val);
                curt = curt.right;
            }
        }
        return result;
    }
}

// 5. Binary Tree Postorder Traversal
public class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> nodeStack = new Stack<TreeNode>();
        Stack<TreeNode> postOrder = new Stack<TreeNode>();
        nodeStack.push(root);
        
        while (!nodeStack.isEmpty()) {
            TreeNode lastNode = nodeStack.pop();
            postOrder.push(lastNode);
            
            if (lastNode.left != null) {
                nodeStack.push(lastNode.left);
            }
            if (lastNode.right != null) {
                nodeStack.push(lastNode.right);
            }
        }
        
        while(!postOrder.isEmpty()) {
            result.add(postOrder.pop().val);
        }
        return result;
    }
}

// 6. Binary Tree Level Order Traversal
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
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (root == null) {
            return result;
        }
        
        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        nodeQueue.offer(root);
        
        while (!nodeQueue.isEmpty()) {
            List<Integer> level = new ArrayList<Integer>();
            int size = nodeQueue.size();
            for (int i = 0; i < size; i++) {
                TreeNode curt = nodeQueue.poll();
                level.add(curt.val);
                
                if (curt.left != null) {
                    nodeQueue.offer(curt.left);
                }
                if (curt.right != null) {
                    nodeQueue.offer(curt.right);
                }
            }
            result.add(level);
        }
        return result;
    }
}
