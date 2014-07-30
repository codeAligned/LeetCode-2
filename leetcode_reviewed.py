__author__ = 'le0nh@rdt'

##################################################
# return the number of different ways to climb stairs, each level can move 4 steps at most.
def climbStairs(num):
    if num <= 4:
        resArray = [0 for i in xrange(5)]
        initArray(resArray)
        return resArray[num]
    elif num >= 5:
        resArray = [0 for i in xrange(num+1)]
        initArray(resArray)
        for i in xrange(5, num+1):
            resArray[i] = resArray[i-1] + resArray[i-2] + resArray[i-3] + resArray[i-4]
        return resArray[num]

def initArray(resArray):
    for i in xrange(len(resArray)):
        resArray[0] = 1
        resArray[1] = 1
        resArray[2] = 2
        resArray[3] = resArray[2] + resArray[1] + resArray[0]
        resArray[4] = resArray[3] + resArray[2] + resArray[1] + resArray[0]
    return resArray

##################################################
# Interview question 1
# change the nth digit of binary value of an integer to 1
def solution(a, offset):
    tmp = a >> (offset - 1)
    if tmp%2 == 1:
        return a
    elif tmp%2 == 0:
        xorVal = 1 << (offset - 1)
        return a^xorVal

##################################################
# Insertion Sort List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a ListNode
    # @return a ListNode
    def insertionSortList(self, head):
        if head == None: return head
        dummy = ListNode(0)
        dummy.next = head
        curr = dummy.next
        while curr.next != None:
            if curr.next.val >= curr.val: curr = curr.next
            else: 
                pre = dummy
                tmp = curr.next
                while pre.next.val < tmp.val:
                    pre = pre.next
                curr.next = tmp.next
                tmp.next = pre.next
                pre.next = tmp

        return dummy.next

##################################################
# Reverse Nodes in k-Group 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a ListNode
    # @param k, an integer
    # @return a ListNode
    def reverseKGroup(self, head, k):
        if head == None: return head
        dummy = ListNode(0)
        dummy.next = head
        start = dummy
        while start.next:
            end = start
            for i in xrange(k-1):
                end = end.next
                if end.next == None: return dummy.next
            
            res = self.reverse(start.next, end.next)
            start.next = res[0]
            start = res[1]
        
        return dummy.next
    
    def reverse(self, start, end):
        _dummy = ListNode(0)
        _dummy.next = start
        
        while _dummy.next != end:
            tmp = start.next
            start.next = tmp.next
            tmp.next = _dummy.next
            _dummy.next = tmp
        
        return [_dummy.next, start]
            