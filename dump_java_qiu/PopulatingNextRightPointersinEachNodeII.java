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
        
        TreeLinkNode dummy = new TreeLinkNode(0);
        TreeLinkNode curt = root;
        TreeLinkNode prev = dummy;
        while (curt != null) {
            if (curt.left != null) {
                prev.next = curt.left;
                prev = prev.next;
            }
            if (curt.right != null) {
                prev.next = curt.right;
                prev = prev.next;
            }
            curt = curt.next;
        }
        connect(dummy.next);
    }
}
