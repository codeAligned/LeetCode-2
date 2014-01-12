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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		ListNode head = new ListNode(0);
		ListNode tmp;
		ListNode cur = head;

		while (true) {
			if (l1 == null) {
				tmp = l2;
				cur.next = tmp;
				return head.next;
			}
			if (l2 == null) {
				tmp = l1;
				cur.next = tmp;
				return head.next;
			}

			if (l1.val < l2.val) {
				tmp = new ListNode(l1.val);
				cur.next = tmp;
				cur = cur.next;
				l1 = l1.next;
			} else {
				tmp = new ListNode(l2.val);
				cur.next = tmp;
				cur = cur.next;
				l2 = l2.next;
			}
		}

	}
}