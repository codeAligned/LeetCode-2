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
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        head = dummy;
        while (head.next != null) {
            ListNode curt = head.next;
            while (curt.next != null && curt.val == curt.next.val) {
                curt = curt.next;
            }
            if (curt != head.next) {
                head.next = curt.next;
            } else {
                head = head.next;
            }
        }
        return dummy.next;
    }
}
