public class Solution {
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return head;
        }
        head = copyNode(head);
        head = copyPtr(head);
        return deleteDup(head);
    }
    
    private RandomListNode copyNode(RandomListNode head) {
        RandomListNode headCopy = head;
        while (head != null) {
            RandomListNode newNode = new RandomListNode(head.label);
            newNode.next = head.next;
            head.next = newNode;
            head = newNode.next;
        }
        return headCopy;
    }
    
    private RandomListNode copyPtr(RandomListNode head) {
        RandomListNode headCopy = head;
        while (head != null && head.next != null) {
            // head.random can be null, need to check null here
            if (head.random != null) {
                head.next.random = head.random.next;
            }
            head = head.next.next;
        }
        return headCopy;
    }
    
    private RandomListNode deleteDup(RandomListNode head) {
        RandomListNode headCopy = head.next;
        while (head != null && head.next != null) {
            RandomListNode temp = head.next;
            // Split the two list, recover the orig list
            head.next = temp.next;
            // set the deep copy list, temp.next can be null
            if (temp.next != null) {
                temp.next = temp.next.next;
            }
            head = head.next;
        }
        return headCopy;
    }
}
