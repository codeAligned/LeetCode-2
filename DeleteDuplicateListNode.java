//Remove Duplicates from Sorted List Total Accepted: 5231 Total Submissions: 15350 My Submissions
//Given a sorted linked list, delete all duplicates such that each element appear only once.
//
//For example,
//Given 1->1->2, return 1->2.
//Given 1->1->2->3->3, return 1->2->3.

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

		if (head != null) {
			ListNode p, q;
			p = head;
			q = p.next;

			while (p != null && p.next != null) {
				if (q.val == p.val) {
					p.next = q.next;
				} else {
					p = p.next;
				}

				q = q.next;
			}
			return head;
		} else {
			return null;
		}
	}	
}