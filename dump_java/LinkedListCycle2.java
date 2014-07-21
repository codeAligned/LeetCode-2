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
    public ListNode detectCycle(ListNode head) {
        
        if (head == null || head.next == null)
			    return null;
		
		ListNode p = head;
		ListNode q = head;

    	p = p.next;
		q = q.next.next;

		while (q != null && q.next != null && !q.equals(p)) {
			p = p.next;
			q = q.next.next;
		}

		if (q == null || q.next == null) {
		    return null;
		} else if(q.equals(p)){
			q = head;
			while (!q.equals(p)) {
				q = q.next;
				p = p.next;
			}
			return q;
		}
    return null;
	
    }
}