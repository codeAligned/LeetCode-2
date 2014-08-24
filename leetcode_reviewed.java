/**
* @author: Yucheng Liu
*/
// ==================================================
// 5. Pass ; 字符串用{}，python用[]
// 6. Help | Pass ; 分左右分别递归
// 7. Pass
// 8. Pass
// 9. Pass
// 10. Pass
// 11. Fail | Pass
// 12. Pass
// 13. 
// ==================================================
/* */


/* 12. Remove Duplicates from Sorted List */
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p = head;
        ListNode q = head.next;
        while (q != null) {
            if (p.val == q.val) {
                p.next = q.next;
                q = q.next;
            } else if (p.val != q.val) {
                p = p.next;
                q = q.next;
            }
        }
        return head;

    }
}


/* 11. Search Insert Position */
public class Solution {
    public int searchInsert(int[] A, int target) {
        
        int p = 0;
        if (target <= A[p]) {
            return p;
        }

        while (p < A.length - 1) {
            if (A[p] == target) {
                return p;
            } else if (A[p] < target && A[p + 1] >= target) {
                return p + 1;
            } else if (A[p] < target && A[p + 1] < target) {
                p++;
            }
        }

        return A.length;
    
    }
}


/* 10. Populating Next Right Pointers in Each Node */
/**
 * Definition for binary tree with next pointer.
 * public class TreeLinkNode {
 *     int val;
 *     TreeLinkNode left, right, next;
 *     TreeLinkNode(int x) { val = x; }
 * }
 */
public class Solution {

    public void connect(TreeLinkNode root) {
        if (root == null) {
            return;
        }

        if (root.left != null) {
            root.left.next = root.right;
        }
        
        if (root.next != null && root.right != null) {
            root.right.next = root.next.left;
        }
        
        connect(root.left);
        connect(root.right);
    }

}


/* 9. Binary Tree Inorder Traversal */
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
        List<Integer> res = new ArrayList<Integer>();
        if( root == null) {
            return res;
        }
        
        if (root.left != null) {
            res.addAll(inorderTraversal(root.left));    
        }
        
        res.add(root.val);
        
        if (root.right != null) {
            res.addAll(inorderTraversal(root.right));    
        }
        
        return res;
    
    }
}


/* 8. Binary Tree Preorder Traversal */
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution8 {

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if( root == null) {
            return res;
        }
        res.add(root.val);
        if (root.left != null) {
            res.addAll(preorderTraversal(root.left));    
        }
        if (root.right != null) {
            res.addAll(preorderTraversal(root.right));    
        }
        
        return res;
    }

}


/* 7. Linked List Cycle */
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        } else if (head.next == null) {
            return false;
        }
        ListNode dummy = new ListNode(0);
        ListNode fast = head;
        ListNode slow = head;
        
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }
}


/* 6. Unique Binary Search Trees */
public class Solution6 {
    public int numTrees(int n) {
        if (n == 0) return 1;
        int res = 0;
        for (int i = 1; i <= n; i++) {
            res += numTrees(n - i) * numTrees(i - 1);
        }
        return res;
    }
}


/* 5. Best Time to Buy and Sell Stock II */
public class Solution5 {
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i + 1] > prices[i]) { // if price increase
                maxProfit += prices[i+1] - prices[i];
            }
        }

        return maxProfit;
    }
}


/* 4. Reverse Integer */
public class Solution4 {
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

/* 3. Same Tree */
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution3 {
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
public class Solution2 {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;    
        }
    }
}


/* 1. Single Number */
public class Solution1 {
    public int singleNumber(int[] A) {
        int ret = 0;
        for(int i = 0; i < A.length; i++) {
            ret ^= A[i];
        }
        
        return ret;
    }
}