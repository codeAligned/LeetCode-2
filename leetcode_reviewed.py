__author__ = 'le0nh@rdt'
# =============== Submit Record ==================
# 21. Pass
# 22. Need help | Pass
# 23. Pass
# 24. Need help | Wrong | Pass
# 25. Fail | Pass
# 26. Pass
# 27. 

# ================================================
#
# 26. Binary Tree Level Order Traversal
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a list of lists of integers
    def levelOrder(self, root):
        if root == None: return []
        self.res = []
        self.dfs(0, root)
        return self.res
        
    def dfs(self, level, root):
        if root == None: return
        if level >= len(self.res):
            self.res.append([root.val])
        else:
            self.res[level].append(root.val)
        self.dfs(level + 1, root.left)
        self.dfs(level + 1, root.right)

#
# 25. Linked List Cycle II 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a ListNode
    # @return a list node
    def detectCycle(self, head):
        if head == None: return None
        dummy = ListNode(0)
        dummy.next = head
        p = dummy; q = dummy
        
        while q.next != None and q.next.next != None:
            p = p.next; q = q.next.next
            if p == q: 
                q = dummy
                while q != p:
                    p = p.next; q = q.next
                return p
        
        return None
#
# 24. Spiral Matrix II 
class Solution:
    # @return a list of lists of integer
    def generateMatrix(self, n):
        res = [[0 for i in xrange(n)] for i in xrange(n)]
        
        count = 1
        # Direction: 0 - Right, 1 - Down, 2 - Left, 3 - Up
        direction = 0
        
        wallUp = -1; wallDown = n; wallLeft = -1; wallRight = n
        
        while count <= n*n:
            if direction == 0:
                for i in xrange(wallLeft + 1, wallRight, 1):
                    res[wallUp + 1][i] = count
                    count += 1
                wallUp += 1
            elif direction == 1:
                for i in xrange(wallUp + 1, wallDown, 1):
                    res[i][wallRight - 1] = count
                    count += 1
                wallRight -= 1
            elif direction == 2:
                for i in xrange(wallRight - 1, wallLeft, -1):
                    res[wallDown - 1][i] = count
                    count += 1
                wallDown -= 1
            elif direction == 3:
                for i in xrange(wallDown - 1, wallUp, -1):
                    res[i][wallLeft + 1] = count
                    count += 1
                wallLeft += 1
            
            direction = (direction + 1) % 4
        
        return res

#
# 23. Binary Tree Postorder Traversal 
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a list of integers
    def postorderTraversal(self, root):
        if root == None: return []
        stack1 = []; stack2 = []
        res = []
        stack1.append(root)
        
        while stack1:
            top = stack1.pop()
            if top.left != None:
                stack1.append(top.left)
            if top.right != None:
                stack1.append(top.right)
            stack2.append(top)
        
        for i in xrange(len(stack2) -1, -1, -1):
            res.append(stack2[i].val)
        return res
#
# 22. Container With Most Water
class Solution:
    # @return an integer
    def maxArea(self, height):
        len_height = len(height)
        if len_height <= 1: return 0
        
        maxVolumn = 0
        start = 0; end = len_height - 1
        while start < end:
            contain = min(height[start], height[end]) * (end - start)
            maxVolumn = max(maxVolumn, contain)
            if height[start] <= height[end]:
                start += 1
            else:
                end -= 1
        
        return maxVolumn


#
# 21. Binary Tree Level Order Traversal II
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a list of lists of integers
    def levelOrderBottom(self, root):
       if root == None: return []
       self.res = []
       self.dfs(0, root)
       self.res.reverse()
       return self.res
      
    def dfs(self, level, root):
        if root == None: return
        if level == len(self.res):
            self.res.append([root.val])
        else:
            self.res[level].append(root.val)
        self.dfs(level+1, root.left)
        self.dfs(level+1, root.right)

#
# 20. Minimum Path Sum


class Solution:
    # @param grid, a list of lists of integers
    # @return an integer

    def minPathSum(self, grid):
        if len(grid) == 0:
            return 0

        m = len(grid)
        n = len(grid[0])

        # initialize first line and first row
        for i in xrange(1, m):
            grid[i][0] += grid[i - 1][0]
        for j in xrange(1, n):
            grid[0][j] += grid[0][j - 1]

        for i in xrange(1, m):
            for j in xrange(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

        return grid[-1][-1]

#
# 19. Best Time to Buy and Sell Stock


class Solution:
    # @param prices, a list of integer
    # @return an integer

    def maxProfit(self, prices):
        if len(prices) < 2:
            return 0

        highestPrice = prices[-1]
        maxProfit = 0

        for i in xrange(len(prices) - 2, -1, -1):
            profit = highestPrice - prices[i]
            maxProfit = max(maxProfit, profit)
            highestPrice = max(highestPrice, prices[i])

        return maxProfit

#
# 18. Search Matrix


class Solution:
    # @param matrix, a list of lists of integers
    # @param target, an integer
    # @return a boolean

    def searchMatrix(self, matrix, target):
        m = len(matrix)
        if m < 1:
            return False

        n = len(matrix[0])
        start = 0
        end = m * n - 1

        while start <= end:
            mid = (start + end) // 2
            i = mid // n
            j = mid % n
            if matrix[i][j] > target:
                end = mid - 1
            elif matrix[i][j] < target:
                start = mid + 1
            else:
                return True
        return False

#
# 17. Permutation


class Solution:
    # @param num, a list of integer
    # @return a list of lists of integers

    def permute(self, num):
        if len(num) == 0:
            []
        if len(num) == 1:
            return [num]
        res = []
        for i in xrange(len(num)):
            for j in self.permute(num[:i] + num[i + 1:]):
                res.append([num[i]] + j)

        return res

#
# 16. Plus One


class Solution:
    # @param digits, a list of integer digits
    # @return a list of integer digits

    def plusOne(self, digits):
        res = [0 for i in xrange(len(digits) + 1)]
        if digits[-1] == 9:
            res[-1] = 0
            carr = 1
        else:
            res[-1] = digits[-1] + 1
            carr = 0

        if len(digits) >= 2:
            for i in xrange(len(digits) - 2, -1, -1):
                if digits[i] + carr > 9:
                    carr = 1
                    res[i + 1] = 0
                else:
                    res[i + 1] = digits[i] + carr
                    carr = 0

        res[0] = carr
        if carr == 0:
            return res[1:]
        else:
            return res

#
# 15. Rotate Image


class Solution:
    # @param matrix, a list of lists of integers
    # @return a list of lists of integers

    def rotate(self, matrix):
        n = len(matrix)
        m = n // 2
        for i in xrange(m):
            for j in xrange(m):
                self.rotateMatrix(i, j, matrix)

        if n % 2 == 1:
            for i in xrange(m):
                j = m
                self.rotateMatrix(i, j, matrix)

        return matrix

    def rotateMatrix(self, i, j, matrix):
        n = len(matrix)
        matrix[i][j], matrix[j][n - 1 - i], matrix[n - 1 - i][n - 1 - j], matrix[n - 1 - j][i] = \
            matrix[n - 1 - j][i], matrix[i][j], matrix[
                j][n - 1 - i], matrix[n - 1 - i][n - 1 - j]

#
# 14. Generate Parenthnese


class Solution:
    # @param an integer
    # @return a list of string

    def generateParenthesis(self, n):
        return self.dfs(0, 0, n)

    def dfs(self, left, right, n):
        if left == n:
            return [")" * (left - right)]
        elif left == right:
            return ["(" + i for i in self.dfs(left + 1, right, n)]
        else:
            return ["(" + i for i in self.dfs(left + 1, right, n)] + [")" + i for i in self.dfs(left, right + 1, n)]

#
# 13. Pascal Triangle


class Solution:
    # @return a list of lists of integers

    def generate(self, numRows):
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1], [1, 1]]

        res = [[1], [1, 1]]
        for i in xrange(2, numRows):
            temp = []
            temp.append(1)
            length = len(res[i - 1])

            for j in xrange(0, length - 1):
                temp.append(res[i - 1][j] + res[i - 1][j + 1])
            temp.append(1)

            res.append(temp)
        return res

#
# 12. Sort Colors


class Solution:
    # @param A a list of integers
    # @return nothing, sort in place

    def sortColors(self, A):
        if len(A) == 1:
            return A
        i = 0
        j = len(A) - 1
        p = 0
        while p <= j:
            if A[p] == 0:
                A[i], A[p] = A[p], A[i]
                i += 1
                p += 1
            elif A[p] == 2:
                A[p], A[j] = A[j], A[p]
                j -= 1
            else:
                p += 1

#
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
                if B[i] == j:
                    valid = False
                    break
                if abs(B[i] - j) == currQ - i:
                    valid = False
                    break
            if valid:
                B[currQ] = j
                self.solve(n, currQ + 1, B)

#
# 10. Merge Tow Sorted Array (correct locally, no idea why online judge
# says wrong answer)


class Solution:
    # @param A  a list of integers
    # @param m  an integer, length of A
    # @param B  a list of integers
    # @param n  an integer, length of B
    # @return nothing

    def merge(self, A, m, B, n):
        if m == 0:
            A += B
        elif n > 0:
            i = 0
            j = 0
            while j < n:
                if B[j] <= A[i]:
                    A.insert(i, B[j])
                    j += 1
                    i += 1
                else:
                    if i >= m + j - 1:
                        A += B[j:]
                        break
                    else:
                        i += 1
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

#
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
        if root == None:
            return True
        return self.dfs(root.left, root.right)

    def dfs(self, leftNode, rightNode):
        if leftNode == None and rightNode == None:
            return True
        elif leftNode != None and rightNode != None:
            if leftNode.val != rightNode.val:
                return False
            else:
                return self.dfs(leftNode.left, rightNode.right) and self.dfs(leftNode.right, rightNode.left)
        else:
            return False

#
# 8. Remove Duplicate from Sorted Array


class Solution:
    # @param a list of integers
    # @return an integer

    def removeDuplicates(self, A):
        if len(A) == 0:
            return 0
        if len(A) == 1:
            return 1

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

#
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
            p = flag.next
            q = p.next
            t = q.next
            flag.next = q
            p.next = q.next
            q.next = p
            p.next = t
            flag = p

        return dummy.next


#
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
        if root == None:
            return True
        return self.checkBalanced(root)[1]

    def checkBalanced(self, root):
        if root == None:
            return (0, True)
        LeftHeight, LeftBool = self.checkBalanced(root.left)
        RightHeight, RightBool = self.checkBalanced(root.right)
        return (max(LeftHeight, RightHeight) + 1, LeftBool and RightBool and abs(LeftHeight - RightHeight) <= 1)


#
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
                curr.next = p
                curr = curr.next
                p = p.next
            else:
                curr.next = q
                curr = curr.next
                q = q.next

        if p == None and q != None:
            curr.next = q
        elif p != None and q == None:
            curr.next = p

        return dummy.next


#
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

#
# 3. Interview question 1
# change the nth digit of binary value of an integer to 1


def solution(a, offset):
    tmp = a >> (offset - 1)
    if tmp % 2 == 1:
        return a
    elif tmp % 2 == 0:
        xorVal = 1 << (offset - 1)
        return a ^ xorVal

#
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

#
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
