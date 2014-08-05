__author__ = 'le0nh@rdt'
##################################################


##################################################


##################################################
# Remove Duplicate from Sorted Array
class Solution:
    # @param a list of integers
    # @return an integer
    def removeDuplicates(self, A):
        if len(A) == 0: return 0
        if len(A) == 1: return 1

        curr = 0
        next = 1
        while next < len(A):
            if A[curr] == A[next]:
                next += 1
            else:
                curr += 1
                A[curr] = A[next]
                next += 1
        return curr + 1

##################################################
# Swap Nodes in Pairs
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param a ListNode
    # @return a ListNode
    def swapPairs(self, head):
        if head == None or head.next == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        
        flag = dummy
        while flag.next and flag.next.next:
            p = flag.next; q = p.next; t = q.next
            flag.next = q
            p.next = q.next
            q.next = p
            p.next = t
            flag = p
            
        return dummy.next
        

##################################################
# Balanced Binary Tree
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a boolean
    def isBalanced(self, root):
        if root == None: return True
        return self.checkBalanced(root)[1]
        
    def checkBalanced(self, root):
        if root == None:
            return (0, True)
        LeftHeight, LeftBool = self.checkBalanced(root.left)
        RightHeight, RightBool = self.checkBalanced(root.right)
        return (max(LeftHeight, RightHeight)+1, LeftBool and RightBool and abs(LeftHeight - RightHeight) <= 1)


##################################################
# Merge Tow Sorted List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param two ListNodes
    # @return a ListNode
    def mergeTwoLists(self, l1, l2):
        dummy = ListNode(0)
        curr = dummy
        p = l1
        q = l2
        while p != None and q != None:
            if p.val < q.val:
                curr.next = p; curr = curr.next
                p = p.next
            else:
                curr.next = q; curr = curr.next
                q = q.next
        
        if p == None and q != None:
            curr.next = q
        elif p != None and q == None:
            curr.next = p
        
        return dummy.next


##################################################
# return the number of different ways to climb stairs, each level can move
# 4 steps at most.


def climbStairs(num):
    if num <= 4:
        resArray = [0 for i in xrange(5)]
        initArray(resArray)
        return resArray[num]
    elif num >= 5:
        resArray = [0 for i in xrange(num + 1)]
        initArray(resArray)
        for i in xrange(5, num + 1):
            resArray[i] = resArray[i - 1] + resArray[i - 2] + \
                resArray[i - 3] + resArray[i - 4]
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
    if tmp % 2 == 1:
        return a
    elif tmp % 2 == 0:
        xorVal = 1 << (offset - 1)
        return a ^ xorVal

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
        if head == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        curr = dummy.next
        while curr.next != None:
            if curr.next.val >= curr.val:
                curr = curr.next
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
        if head == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        start = dummy
        while start.next:
            end = start
            for i in xrange(k - 1):
                end = end.next
                if end.next == None:
                    return dummy.next

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
