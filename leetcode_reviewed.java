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
// 13. Pass
// 14. Help | Pass | bit operation
// 15. Help | Pass | two flag value - sum and max
// 16. Pass | Simplest way to print array - Arrays.toString(/*some int[] or array[] value*/)
// 17. Fail | Pass | didn't return after currQueen == n (again! same mistake!!)
// 18. Help | Pass | subarray in java: Arrays.copyOfRange(src, 0, 2);
// 19. Pass
// 20. Fail | Pass; (Again! should not put ss.empty() in while condition)
// ==================================================
/* 20. Valid Parentheses */
public class Solution20 {
    public boolean isValid(String s) {

        if (s.length() <= 1)
            return false;

        HashMap<Character, Character> ht = new HashMap<Character, Character>();
        ht.put('(', ')');
        ht.put('[', ']');
        ht.put('{', '}');

        Stack<Character> ss = new Stack<Character>();
        if (ht.containsKey(s.charAt(0)))
            ss.push(s.charAt(0));
        else
            return false;

        int i = 1;
        while (i < s.length()) {
            if (ht.containsKey(s.charAt(i))) {
                ss.push(s.charAt(i));
                i++;
            } else {
                if (ss.empty() || s.charAt(i) != ht.get(ss.pop())) {
                    return false;
                } else {
                    i++;
                }
            }
        }

        if (i == s.length() && ss.empty())
            return true;
        else
            return false;
    }
}


/* 19. Balanced Binary Tree */
public class Solution19 {
    public boolean isBalanced(TreeNode root) {
        if (root == null)
            return true;
        // if (root.left == null && root.right == null) return true;
        return isBalanced(root.left) && isBalanced(root.right) && Math.abs(getHeight(root.left) - getHeight(root.right)) <= 1;
    }

    public int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        } else if (root.left == null && root.right == null) {
            return 1;
        } else if (root.left != null && root.left != null) {
            return Math.max(getHeight(root.left), getHeight(root.right)) + 1;
        } else if (root.left != null) {
            return getHeight(root.left) + 1;
        } else {
            return getHeight(root.right) + 1;
        }
    }
}


/* 18. Convert Sorted Array to Binary Search Tree */
public class Solution18 {
    public TreeNode sortedArrayToBST(int[] num) {
        if (num.length <= 0)
            return null;

        int mid = num.length / 2;
        TreeNode root = new TreeNode(num[mid]);

        if (mid > 0)
            root.left = sortedArrayToBST(Arrays.copyOfRange(num, 0, mid));
        if (mid < num.length - 1)
            root.right = sortedArrayToBST(Arrays.copyOfRange(num, mid + 1,
                    num.length));

        return root;
    }
}


/* 17. N-Queens II */ 
public class Solution17 {
    public int res = 0;

    public int totalNQueens(int n) {
        int[] board = new int[n];
        for (int i = 0; i < n; i++) {
            board[i] = -1;
        }

        dps(0, board, n);

        return res;

    }

    public void dps(int currQueen, int[] board, int n) {
        if (currQueen == n) {
            res++;
            return;
        }
        for (int i = 0; i < n; i++) {
            boolean isValid = true;
            for (int k = 0; k < currQueen; k++) {
                if (board[k] == i) {
                    isValid = false;
                    break;
                } 
                if ((Math.abs(board[k] - i)) == (currQueen - k)) {
                    isValid = false;
                    break;
                }
            }
            if (isValid) {
                board[currQueen] = i;
                dps(currQueen + 1, board, n);
            }
        }
    }
}


/* 16. Remove Element */
public class Solution16 {
    public int removeElement(int[] A, int elem) {
        int p = 0, q = A.length-1;
        int tmp = 0;
        while (p <= q) {
            if (A[q] == elem) {
                q--;
            } else if (A[p] == elem) {
                tmp = A[q];
                A[q] = A[p];
                A[p] = tmp;
                p++;
                q--;
            } else {
                p++;
            }

        }
        return p;
    }
}


/* 15. Maximum Subarray*/
public class Solution15 {
    public int maxSubArray(int[] A) {
        int max = Integer.MIN_VALUE;
        int sum = 0;
        for (int i = 0; i < A.length; i++) {
            if (sum < 0) sum = 0;
            sum += A[i];
            max = Math.max(sum, max);
        }

        return max;
    }
}


/* 14. Single Number II */
public class Solution14 {
    public int singleNumber(int[] A) {      
        int res = 0;
        for (int i = 0; i < 32; ++i) {// assume dealing with int32.
            int bit = 0;
            for (int j = 0; j < A.length; ++j) {
                bit = (bit + ((A[j] >> i) & 1)) % 3;
            }
            res += (bit << i);
        }
        return res;
    }
}


/* 13. Climb Stairs*/
public class Solution13 {
    public int climbStairs(int n) {
        if (n <= 1) {
            return 1;
        }
        int[] res = new int[n + 1];
        res[0] = 1;
        res[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            res[i] = res[i - 1] + res[i - 2];
        }

        return res[n];
    }
}


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
public class Solution12 {
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
public class Solution11 {
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
public class Solution10 {

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
public class Solution9 {
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