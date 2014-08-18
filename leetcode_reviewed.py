__author__ = 'le0nh@rdt'
# =============== Submit Record ==================
# 45. Pass
# 46. Help | Pass
# 47. Pass
# 48. Help | Fail | Pass
# 49. Fail | Pass
# 50. Pass
# 51. Fail | Pass
# 52. Fail | Fail | Fail | Pass
# 53. Fail | Fail | Pass
# 54. Help | Fail | Fail | Pass
# 55. Fail | Fail | Pass
# 56. Help | Fail | Pass
# 57. Help | Pass
# 58. Fail | Pass
# ================================================
# 58. Combinations
class Solution:
    # @return a list of lists of integers
    def combine(self, n, k):
        A = [i for i in xrange(1, n + 1)]
        return self.dfs(k, A)

    def dfs(self, k, A):
        if k == 1:
            ret = []
            for i in A:
                ret.append([i])
            return ret
        elif k == len(A):
            return [A]
        else:
            ret = []
            for i in xrange(0, len(A)):
                for j in self.dfs(k - 1, A[i+1:]):
                    ret.append([A[i]]+j)
            return ret


# 57. Remove Duplicates from Sorted Array II 
class Solution:
    # @param A a list of integers
    # @return an integer
    def removeDuplicates(self, A):
        if len(A) <= 2: return len(A)
        slow = 0; count = 1
        for fast in xrange(1, len(A)):
            if A[fast] == A[slow]:
                count += 1
                if count <= 2:
                    slow += 1
                    A[slow] = A[fast]
            else:
                count = 1
                slow += 1
                A[slow] = A[fast]
        
        return slow + 1

#
# 56. Edit Distance
class Solution:
    # @return an integer
    def minDistance(self, word1, word2):
        len1 = len(word1); len2 = len(word2)
        if len(word1) == 0: return len2
        if len(word2) == 0: return len1
        
        dp = [[0 for j in xrange(len2 + 1)] for i in xrange(len1 + 1)]
        
        for i in xrange(1, len1 + 1):
            dp[i][0] = i
        for j in xrange(1, len2 + 1):
            dp[0][j] = j
        
        for i in xrange(1, len1 + 1):
            for j in xrange(1, len2 + 1):
                if word1[i-1] == word2[j-1]: 
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        return dp[len1][len2]

#
# 55. Valid Parenthese
class Solution:
    # @return a boolean
    def isValid(self, s):
        right = ")]}"
        left = "([{"
        if len(s) == 0: return True
        if len(s) == 1: return False
        
        if s[0] in right: return False
        
        dict = {'(':')', '[':']', '{':'}'}
        stack = []
        stack.append(s[0])
        i = 1
        
        while i < len(s):
            if s[i] in right:
                if stack:
                    tmp = stack.pop()
                    if dict[tmp] != s[i]:
                        return False
                else: return False
            else:
                stack.append(s[i])
            i += 1
        
        if not stack and i == len(s):
            return True
        else:
            return False

#
# 54. Populating Next Right Pointer in Each Node II
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree node
    # @return nothing

    def connect(self, root):
        if root == None:
            return None
        curr = root
        nexthead = None
        nextend = None

        while curr:
            if curr.left:
                if nexthead:
                    nextend.next = curr.left
                    nextend = nextend.next
                else:
                    nexthead = curr.left
                    nextend = nexthead
            if curr.right:
                if nexthead:
                    nextend.next = curr.right
                    nextend = nextend.next
                else:
                    nexthead = curr.right
                    nextend = nexthead

            if curr.next == None:
                curr = nexthead
                nexthead = None
                nextend = None
            else:
                curr = curr.next


#
# 53. LRU Cache
class LRUCache:

    # @param capacity, an integer

    def __init__(self, capacity):
        self.dict = collections.OrderedDict()
        self.capacity = capacity
        self.numItems = 0

    # @return an integer
    def get(self, key):
        try:
            value = self.dict[key]
            self.set(key, value)
            return value
        except:
            return -1

    # @param key, an integer
    # @param value, an integer
    # @return nothing
    def set(self, key, value):
        try:
            del self.dict[key]
            self.dict[key] = value
        except:
            if self.numItems == self.capacity:
                self.dict.popitem(last=False)
                self.numItems -= 1
            self.dict[key] = value
            self.numItems += 1
        return
#
# 52. Path Sum
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    # @param root, a tree node
    # @param sum, an integer
    # @return a boolean

    def hasPathSum(self, root, sum):
        if root == None:
            return False
        return self.dfs(sum, root)

    def dfs(self, target, root):
        if root.left == None and root.right == None:
            if target == root.val:
                return True
            else:
                return False

        left = False
        right = False

        if root.left != None:
            left = self.dfs(target - root.val, root.left)
        if root.right != None:
            right = self.dfs(target - root.val, root.right)

        return left or right
#
# 51. Set Matrix Zeroes


class Solution:
    # @param matrix, a list of lists of integers
    # RETURN NOTHING, MODIFY matrix IN PLACE.

    def setZeroes(self, matrix):
        if len(matrix) == 0:
            return

        m = len(matrix)
        n = len(matrix[0])
        firstColumn = False
        firstRow = False

        for i in xrange(0, m):
            if matrix[i][0] == 0:
                firstColumn = True
                continue
        for j in xrange(0, n):
            if matrix[0][j] == 0:
                firstRow = True
                continue

        for i in xrange(1, m):
            for j in xrange(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        for i in xrange(1, m):
            if matrix[i][0] == 0:
                for j in xrange(1, n):
                    matrix[i][j] = 0
        for j in xrange(1, n):
            if matrix[0][j] == 0:
                for i in xrange(1, m):
                    matrix[i][j] = 0

        if firstColumn:
            for i in xrange(m):
                matrix[i][0] = 0
        if firstRow:
            for j in xrange(n):
                matrix[0][j] = 0


#
# 50. Binary Tree Level Order Traversal
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
        if root == None:
            return []
        self.res = []
        self.dfs(0, root)
        return self.res

    def dfs(self, level, root):
        if root == None:
            return
        if level >= len(self.res):
            self.res.append([root.val])
        else:
            self.res[level].append(root.val)
        self.dfs(level + 1, root.left)
        self.dfs(level + 1, root.right)

#
# 49. Linked List Cycle II
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    # @param head, a ListNode
    # @return a list node

    def detectCycle(self, head):
        if head == None:
            return None
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        q = dummy

        while q.next != None and q.next.next != None:
            p = p.next
            q = q.next.next
            if p == q:
                q = dummy
                while q != p:
                    p = p.next
                    q = q.next
                return p

        return None
#
# 48. Spiral Matrix II


class Solution:
    # @return a list of lists of integer

    def generateMatrix(self, n):
        res = [[0 for i in xrange(n)] for i in xrange(n)]

        count = 1
        # Direction: 0 - Right, 1 - Down, 2 - Left, 3 - Up
        direction = 0

        wallUp = -1
        wallDown = n
        wallLeft = -1
        wallRight = n

        while count <= n * n:
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
# 47. Binary Tree Postorder Traversal
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
        if root == None:
            return []
        stack1 = []
        stack2 = []
        res = []
        stack1.append(root)

        while stack1:
            top = stack1.pop()
            if top.left != None:
                stack1.append(top.left)
            if top.right != None:
                stack1.append(top.right)
            stack2.append(top)

        for i in xrange(len(stack2) - 1, -1, -1):
            res.append(stack2[i].val)
        return res
#
# 46. Container With Most Water


class Solution:
    # @return an integer

    def maxArea(self, height):
        len_height = len(height)
        if len_height <= 1:
            return 0

        maxVolumn = 0
        start = 0
        end = len_height - 1
        while start < end:
            contain = min(height[start], height[end]) * (end - start)
            maxVolumn = max(maxVolumn, contain)
            if height[start] <= height[end]:
                start += 1
            else:
                end -= 1

        return maxVolumn


#
# 45. Binary Tree Level Order Traversal II
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
        if root == None:
            return []
        self.res = []
        self.dfs(0, root)
        self.res.reverse()
        return self.res

    def dfs(self, level, root):
        if root == None:
            return
        if level == len(self.res):
            self.res.append([root.val])
        else:
            self.res[level].append(root.val)
        self.dfs(level + 1, root.left)
        self.dfs(level + 1, root.right)

#
# 44. Minimum Path Sum


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
# 43. Best Time to Buy and Sell Stock


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
# 42. Search Matrix


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
# 41. Permutation


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
# 40. Plus One


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
# 39. Rotate Image


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
# 38. Generate Parenthnese


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
# 37. Pascal Triangle


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
# 36. Sort Colors


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
# 35. N Queens II


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
# 34. Merge Tow Sorted Array (correct locally, no idea why online judge
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
# 33. Symmetric Tree
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
# 32. Remove Duplicate from Sorted Array


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
# 31. Swap Nodes in Pairs
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
# 30. Balanced Binary Tree
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
# 29. Merge Tow Sorted List
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
# 28. Insertion Sort List
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
# 27. Reverse Nodes in k-Group
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
