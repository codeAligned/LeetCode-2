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

// 7. Subsets
public class Solution {
    public List<List<Integer>> subsets(int[] S) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (S == null || S.length == 0) {
            return result;
        }
        
        Arrays.sort(S);
        List<Integer> solution = new ArrayList<Integer>();
        helpGetSubsets(S, 0, solution, result);
        return result;
    }
    
    private void helpGetSubsets(int[] S, int start, List<Integer> solution,
        List<List<Integer>> result) {
        result.add(new ArrayList<Integer>(solution));
        
        for (int i = start; i < S.length; i++) {
            solution.add(S[i]);
            helpGetSubsets(S, i + 1, solution, result);
            solution.remove(solution.size() - 1);
        }
    }
}

// 8. Subsets II
public class Solution {
    public List<List<Integer>> subsetsWithDup(int[] num) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (num == null || num.length == 0) {
            return result;
        }
        
        Arrays.sort(num);
        List<Integer> combination = new ArrayList<Integer>();
        helpGetSubsets(num, 0, combination, result);
        return result;
    }
    
    private void helpGetSubsets(int[] num, int start, List<Integer> combination, List<List<Integer>> result) {
        result.add(new ArrayList<Integer>(combination));
        
        for (int i = start; i < num.length; i++) {
            if (i > start && num[i] == num[i - 1]) {
                continue;
            }
            combination.add(num[i]);
            helpGetSubsets(num, i + 1, combination, result);
            combination.remove(combination.size() - 1);
        }
    }
}

// 9. Permutations
public class Solution {
    public List<List<Integer>> permute(int[] num) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (num == null || num.length == 0) {
            return result;
        }
        List<Integer> permutation = new ArrayList<Integer>();
        boolean[] visited = new boolean[num.length];
        helpPermute(num, permutation, visited, result);
        return result;
    }
    
    public void helpPermute(int[] num, List<Integer> permutation, boolean[] visited, 
        List<List<Integer>> result) {
        
        if (permutation.size() == num.length) {
            result.add(new ArrayList<Integer>(permutation));
            return;
        }
        
        for (int i = 0; i < num.length; i++) {
            if (!visited[i]) {
                permutation.add(num[i]);
                visited[i] = true;
                helpPermute(num, permutation, visited, result);
                visited[i] = false;
                permutation.remove(permutation.size() - 1);
            }
        }
    }
}

// 10. Permutations II
public class Solution {
    public List<List<Integer>> permuteUnique(int[] num) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (num == null || num.length == 0) {
            return result;
        }
        boolean[] visited = new boolean[num.length];
        List<Integer> list = new ArrayList<Integer>();
        // first sort the array
        Arrays.sort(num);
        helpPermute(num, visited, list, result);
        return result;
    }
    
    public void helpPermute(int[] num, boolean[] visited, List<Integer> list,
    List<List<Integer>> result) {
        if (list.size() == num.length) {
            result.add(new ArrayList<Integer>(list));
            return;
        }
        for (int i = 0; i < num.length; i++) {
            // Do not consider a duplicated number if its earlier appearance has
            // not been considered yet
            if (visited[i] || (i != 0 && num[i] == num[i - 1] && !visited[i - 1])) {
                continue;
            }
            list.add(num[i]);
            visited[i] = true;
            helpPermute(num, visited, list, result);
            list.remove(list.size() - 1);
            visited[i] = false;
        }
    }
}

// 11. Combinations
public class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (k <= 0 || n <= 0) {
            return result;
        }
        List<Integer> combination = new ArrayList<Integer>();
        getCombination(combination, k, n, 1, result);
        return result;
    }
    
    private void getCombination(List<Integer> combination, int k, int n, int start, 
        List<List<Integer>> result) {
        if (combination.size() == k) {
            result.add(new ArrayList<Integer>(combination));
            return;
        }
        
        for (int i = start; i <= n ; i++) {
            combination.add(i);
            getCombination(combination, k, n, i + 1, result);
            combination.remove(combination.size() - 1);
        }
    }
}
