__author__ = 'le0nh@rdt'
##################################################
# 12. Sort Colors
class Solution:
    # @param A a list of integers
    # @return nothing, sort in place
    def sortColors(self, A):
        if len(A) == 1: return A
        i = 0; j = len(A)-1
        p = 0
        while p <= j:
            if A[p] == 0:
                A[i], A[p] = A[p], A[i]
                i += 1; p += 1
            elif A[p] == 2:
                A[p], A[j] = A[j], A[p]
                j -= 1
            else:
                p += 1

##################################################
# 11. N Queens II
class Solution:
    # @return an integer
    def totalNQueens(self, n):
        self.res = 0
        self.solve(n, 0, [-1 for i in xrange(n)])
        return self.res
        
    def solve(self, n, currQ, B):
        if currQ == n:
            self.res += 1
            return
        for j in xrange(n):
            valid = True
            for i in xrange(currQ):
                if B[i] == j: valid = False; break
                if abs(B[i] - j) == currQ - i: valid = False; break
            if valid:
                B[currQ] = j
                self.solve(n, currQ + 1, B)

##################################################
# 10. Merge Tow Sorted Array (correct locally, no idea why online judge says wrong answer)
class Solution:
    # @param A  a list of integers
    # @param m  an integer, length of A
    # @param B  a list of integers
    # @param n  an integer, length of B
    # @return nothing
    def merge(self, A, m, B, n):
        if m == 0: A += B
        elif n > 0:
            i = 0; j = 0
            while j < n:
                if B[j] <= A[i]:
                    A.insert(i, B[j])
                    j += 1; i+= 1
                else:
                    if i >= m+j-1:
                        A += B[j:]; break
                    else: i += 1
# 从后往前排
class Solution:
    # @param A  a list of integers
    # @param m  an integer, length of A
    # @param B  a list of integers
    # @param n  an integer, length of B
    # @return nothing
    def merge(self, A, m, B, n):
        i, j, k = m - 1, n - 1, m + n - 1
        while i >= 0 and j >= 0:
            if B[j] > A[i]:
                A[k] = B[j]
                j -= 1
            else:
                A[k] = A[i]
                i -= 1
            k -= 1
        while j >= 0:
            A[k] = B[j]
            j -= 1
            k -= 1

##################################################
# 9. Symmetric Tree
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a boolean
    def isSymmetric(self, root):
        if root == None: return True
        return self.dfs(root.left, root.right)
    
    def dfs(self, leftNode, rightNode):
        if leftNode == None and rightNode == None: return True
        elif leftNode != None and rightNode != None:
            if leftNode.val != rightNode.val: return False
            else: return self.dfs(leftNode.left, rightNode.right) and self.dfs(leftNode.right, rightNode.left)
        else: return False

##################################################
# 8. Remove Duplicate from Sorted Array
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
# 7. Swap Nodes in Pairs
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
# 6. Balanced Binary Tree
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
# 5. Merge Tow Sorted List
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
# 4. return the number of different ways to climb stairs, each level can move
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
# 3. Interview question 1
# change the nth digit of binary value of an integer to 1


def solution(a, offset):
    tmp = a >> (offset - 1)
    if tmp % 2 == 1:
        return a
    elif tmp % 2 == 0:
        xorVal = 1 << (offset - 1)
        return a ^ xorVal

##################################################
# 2. Insertion Sort List
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
# 1. Reverse Nodes in k-Group
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
