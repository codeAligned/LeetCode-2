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
		if (head == null)
			return null;

		ListNode p = head;
		ListNode q = p;

		if (p.next != null && p.next.next != null) {
			q = p.next;
			p.next = q.next;
			q.next = p;
			head = q;
			q = p;
		} else if (p.next != null && p.next.next == null) {
			q = p.next;
			p.next = q.next;
			q.next = p;
			head = q;
		}

		while(p != null){
			if(p.next == null){
				break;
			}else if(p.next != null && p.next.next != null){
				q = p.next.next;
				p.next.next = q.next;
				q.next = p.next;
				p.next = q;
				q = q.next;
				p = q;
			}else if (p.next != null && p.next.next == null){
				break;
			}
		}
		
		return head;
	}
}