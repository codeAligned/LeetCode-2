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
        
        TreeLinkNode parent = root;
        while (parent != null) {
            TreeLinkNode next = null;
            TreeLinkNode prev = null;
            while (parent != null) {
                if (prev == null) {
                    if (parent.left == null) {
                        break;
                    } else {
                        prev = parent.left;
                        next = prev;
                    }
                } else {
                    prev.next = parent.left;
                    prev = prev.next;
                }
                prev.next = parent.right;
                prev = prev.next;
                parent = parent.next;
            }
            parent = next;
        }
    }
}


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
        buildConnect(root, null);
    }
    
    public void buildConnect(TreeLinkNode root, TreeLinkNode next) {
        if (root == null) {
            return;
        }
        root.next = next;
        buildConnect(root.left, root.right);
        buildConnect(root.right, root.next == null? null : root.next.left);
    }
}
