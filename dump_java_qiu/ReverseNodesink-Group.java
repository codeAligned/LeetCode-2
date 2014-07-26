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
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k < 2) {
            return head;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode preNode = dummy;
        
        int i = 0;
        while (head != null) {
            i++;
            if (i % k != 0) {
                head = head.next;
            } else {
                preNode = reverse(preNode, head.next);
                head = preNode.next;
            }
        }
        return dummy.next;
    }
    
    public ListNode reverse(ListNode preNode, ListNode next) {
        ListNode curt = preNode.next;
        while (curt.next != next) {
            ListNode temp = curt.next;
            curt.next = temp.next;
            temp.next = preNode.next;
            preNode.next = temp;
        }
        return curt;
    }
}
