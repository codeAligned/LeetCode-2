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
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(-1);
        dummy.next = head;

        ListNode pre = dummy;
        while (pre.next != null && pre.next.next != null) {
            ListNode first = pre.next;
            ListNode second = first.next;
            ListNode next = second.next;
            pre.next = second;
            second.next = first;
            first.next = next;
            pre = first;
        }
        return dummy.next;
    }
}
