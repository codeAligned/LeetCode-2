/**
* @author: Yucheng Liu
*/

/* 4. Reverse Integer */
public class Solution {
    public int reverse(int x) {
        if (x == 0) return 0;
        
        boolean notNegtive = false;
        
        if (x >= 0) {
            notNegtive = true;
        } else {
            notNegtive = false;
            x = -x;
        }
        
        int ret = getreverse(x);
        if (notNegtive) {
            return ret;
        } else {
          return -ret;  
        }
    }

    public int getreverse(int x) {
        int remainder;
        int ret = 0;
        while(x > 0) {
            remainder = x % 10;
            x = x / 10;
            ret = ret * 10 + remainder;
        }
        return ret;
    }
}

/* 3. Same Tree*/
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
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p != null && q != null) {
            return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        } else {
            return false;
        }
    }
}

/* 2. Maximum Depth of Binary Tree */
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
        } else {
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;    
        }
    }
}


/* 1. Single Number */
public class Solution {
    public int singleNumber(int[] A) {
        int ret = 0;
        for(int i = 0; i < A.length; i++) {
            ret ^= A[i];
        }
        
        return ret;
    }
}