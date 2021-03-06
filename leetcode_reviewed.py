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
# 59. Pass
# 60. Fail | Fail | Pass
# 61. Pass
# 62. Pass
# 63. Help | Pass
# 64. Help | Fail
# 65. Help | Pass
# 66. Pass
# 67. Help | Pass ; use HashMap/Dictionary
# 68. Help | Fail | Pass ; use temprary array to record duplicates
# 69. Pass
# 70. Help | Help ; !!! Review, dfs !!!
# 71. Fail | Fail | Pass
# 72. Pass
# 73. Help | Pass
# 74. Help | Pass 
# 75. Fail | Pass
# 76. Help | Pass ; use value farest to record
# 77. Help | Pass ; no easy way, do bruteforece, but sort at first and search start at first and last.
# 78. Fail | Pass ; root value can be negtive!
# 79. Fail | Pass
# 80. Fail | Pass
# 81. Help | Fail | Pass ; use self.pre to track the previous node to current root, compare their value.
# 82. Pass
# 83. Pass
# 84. Pass
# 85. Fail | Pass
# 86. Fail | Pass ; see the commented line, should not use "if not in visited"
# 87. TLE  | Pass ; cutting branches - if target < candidates[i]: return
# 88. Fail | Pass ; misunderstand the quesiton, but method is correct
# 89. Help ; Trick is insert cloned list node to original list one by one and extract these nodes
# 90. Help | Pass ; Trick is need to stacks, one for height, one for index
# 91. Pass
# 92. Help | TLE : Trick is use fast ans slower pointer to seperate the original list into two list and do merge sort

# ================================================
# 92. Sort List
class Solution:
    # @param head, a ListNode
    # @return a ListNode
    def sortList(self, head):
        return self.quicksort(head)

    def quicksort(self, head):
        if head is None or head.next is None:
            return head

        ptr = head.next;
        head.next = None
        smallptr = ListNode(0);
        dummy_small = smallptr
        largeptr = ListNode(0);
        dummy_large = largeptr

        while ptr is not None:
            if ptr.val < head.val:
                smallptr.next = ptr
                smallptr = smallptr.next
            elif ptr.val > head.val:
                largeptr.next = ptr
                largeptr = largeptr.next

            ptr = ptr.next

        smallptr.next = None
        largeptr.next = None

        newhead = self.quicksort(dummy_small.next)
        if newhead is None:
            newhead = head
        else:
            tmp = newhead
            while tmp.next is not None:
                tmp = tmp.next
            tmp.next = head  # how to get the tail of small half???

        head.next = self.quicksort(dummy_large.next)

        return newhead


# 91. Maximal Rectangle
class Solution:
    # @param matrix, a list of lists of 1 length string
    # @return an integer
    def maximalRectangle(self, matrix):
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[0])):
                if matrix[i][j] == '1': matrix[i][j] = 1
                else: matrix[i][j] = 0
        for i in xrange(1, len(matrix)):
            for j in xrange(len(matrix[0])):
                if matrix[i][j] != 0:
                    matrix[i][j] += matrix[i-1][j]
        maximalRec = 0
        for height in matrix:
            maximalRec = max(maximalRec, self.largestRectangleArea(height))
        return maximalRec
    
    def largestRectangleArea(self, height):
        heightstack = [0]
        indexstack = [0]
        maxrec = 0
        for i in range(len(height)):
            if (not heightstack) or height[i] > heightstack[-1]:
                heightstack.append(height[i])
                indexstack.append(i + 1)
            elif heightstack and height[i] < heightstack[-1]:
                while height[i] < heightstack[-1]:
                    maxheight = heightstack.pop()
                    lastindex = indexstack.pop()
                    tmprec = (i + 1 - lastindex) * maxheight
                    maxrec = max(tmprec, maxrec)
                heightstack.append(height[i])
                indexstack.append(lastindex)

        curr = len(height) + 1
        while heightstack:
            maxheight = heightstack.pop()
            lastindex = indexstack.pop()
            tmprec = (curr - lastindex) * maxheight
            maxrec = max(tmprec, maxrec)

        return maxrec

# 90. Largest Rectangle in Histogram
class Solution:
    # @param height, a list of integer
    # @return an integer
    def largestRectangleArea(self, height):
        heightstack = [0]
        indexstack = [0]
        maxrec = 0
        for i in range(len(height)):
            if (not heightstack) or height[i] > heightstack[-1]:
                heightstack.append(height[i])
                indexstack.append(i + 1)
            elif heightstack and height[i] < heightstack[-1]:
                while height[i] < heightstack[-1]:
                    maxheight = heightstack.pop()
                    lastindex = indexstack.pop()
                    tmprec = (i + 1 - lastindex) * maxheight
                    maxrec = max(tmprec, maxrec)
                heightstack.append(height[i])
                indexstack.append(lastindex)

        curr = len(height) + 1
        while heightstack:
            maxheight = heightstack.pop()
            lastindex = indexstack.pop()
            tmprec = (curr - lastindex) * maxheight
            maxrec = max(tmprec, maxrec)

        return maxrec


# 89. Copy List with Random Pointer
# Definition for singly-linked list with a random pointer.
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution:
    # @param head, a RandomListNode
    # @return a RandomListNode
    def copyRandomList(self, head):
        if head == None:
            return None
        p = head
        count = 0
        while p:
            # create a new node and insert between p and p.next
            t = RandomListNode(0)
            t.next = p.next
            t.random = p.random
            t.label = p.label
            p.next = t
            p = t.next
            count += 1

        p = head.next

        for i in xrange(count):
            if p.random:
                p.random = p.random.next
            if p and p.next:
                p = p.next.next

        p1, p2, newHead = head, head.next, head.next
        for i in xrange(count - 1):
            p1.next, p2.next = p1.next.next, p2.next.next
            p1, p2 = p1.next, p2.next
        p1.next, p2.next = None, None

        return newHead


# 88. Anagrams
class Solution:
    # @param strs, a list of strings
    # @return a list of strings
    def anagrams(self, strs):
        if len(strs) <= 1:
            return []
        
        dict = {}
        for s in strs:
            key = ''.join(sorted(s))
            if not dict.has_key(key):
                dict[key] = [s]
            else:
                dict[key].append(s)

        res = []
        for k in dict:
            if len(dict[k]) > 1:
                res += dict[k]

        return res
# 87. Combination Sum II 
class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum2(self, candidates, target):
        candidates.sort()
        lencandidates = len(candidates)
        res = []

        def dfs(start, target, valuelist):
            if target < 0:
                return
            elif target == 0:
                if valuelist not in res:
                    res.append(valuelist)
                return

            for i in xrange(start, lencandidates):
                if target < candidates[i]:
                    return
                dfs(i + 1, target - candidates[i], valuelist + [candidates[i]])

        dfs(0, target, [])
        return res


# 86. Clone Graph
class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        self.visited = {}
        
        def dfs(node):
            if node is None:
                return node
            if self.visited.has_key(node):
                return self.visited[node]
            else:
                clone = UndirectedGraphNode(node.label)
                self.visited[node] = clone
                for neighbor in node.neighbors:
                    #if neighbor not in self.visited:
                    clone.neighbors.append(dfs(neighbor))
            return clone

        return dfs(node)


# 85. Add Binary
class Solution:
    # @param a, a string
    # @param b, a string
    # @return a string
    def addBinary(self, a, b):
        a = a[::-1]
        b = b[::-1]

        length = max(len(a), len(b))
        a += '0' * (length - len(a))
        b += '0' * (length - len(b))

        res = [0 for i in xrange(length + 1)]
        for i in xrange(length):
            res[i] = int(a[i]) + int(b[i])

        carry = 0
        for i in xrange(length + 1):
            res[i] += carry
            carry = res[i] // 2
            res[i] = res[i] % 2

        end = length
        while end > 0:
            if res[end] == 0:
                end -= 1
            else:
                break
        return ''.join(str(s) for s in res[end::-1])


# 84. Letter Combinations of a Phone Number 
class Solution:
    # @return a list of strings, [s1, s2]
    def letterCombinations(self, digits):
        dict = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        self.res = [""]

        def dfs(digits, valuelist):
            if digits == '':
                # self.res += valuelist
                self.res = valuelist
                return
            tmp = []
            if len(valuelist) == 0:
                for c in dict[digits[0]]:
                    tmp.append(c)
            else:
                for c in dict[digits[0]]:
                    for s in valuelist:
                        tmp.append(s + c)
            dfs(digits[1:], tmp)

        dfs(digits, [""])
        return self.res



# 83. Combination Sum
class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        lenCandidates = len(candidates)
        def dfs(start, target, valuelist):
            if target == 0:
                res.append(valuelist)
            elif target < 0:
                return
            for i in xrange(start, lenCandidates):
                dfs(i, target - candidates[i], valuelist + [candidates[i]])

        dfs(0, target, [])
        return res


# 82. Triangle
class Solution:
    # @param triangle, a list of lists of integers
    # @return an integer
    def minimumTotal(self, triangle):
        if len(triangle) == 1:
            return triangle[0][0]

        for ii in range(1, len(triangle)):
            triangle[ii][0] += triangle[ii-1][0]
            for jj in range(1, len(triangle[ii]) - 1):
                triangle[ii][jj] += min(triangle[ii-1][jj-1], triangle[ii-1][jj])
            triangle[ii][-1] += triangle[ii-1][-1]

        return min(triangle[-1])


# 81. Recover Binary Search Tree
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a tree node
    def recoverTree(self, root):
        self.first = None
        self.second = None
        self.pre = TreeNode(-10000)
        self.inorder(root)

        self.first.val, self.second.val = self.second.val, self.first.val

        return root

    def inorder(self, root):
        if root is not None:
            self.inorder(root.left)

            if root.val < self.pre.val:
                if self.first is None:
                    self.first = self.pre
                self.second = root
            self.pre = root

            self.inorder(root.right)


# 80. Partition List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    # @param head, a ListNode
    # @param x, an integer
    # @return a ListNode
    def partition(self, head, x):
        if head is None: return head

        dummy1 = ListNode(0); p1 = dummy1
        dummy2 = ListNode(1); p2 = dummy2
        ptr = head

        while ptr is not None:
            if ptr.val < x:
                p1.next = ptr
                p1 = p1.next
                ptr = p1.next
            else:
                p2.next = ptr
                p2 = p2.next
                ptr = p2.next

        p1.next = dummy2.next
        p2.next = None

        return dummy1.next


# 79. Subsets II
class Solution:
    # @param num, a list of integer
    # @return a list of lists of integer
    def subsetsWithDup(self, S):
        res = []
        lenS = len(S)
        def dfs(depth, start, valuelist):
            if valuelist not in res:
                res.append(valuelist)
            if depth == len(S):
                return
            for i in xrange(start, lenS):
                dfs(depth + 1, i + 1, valuelist + [S[i]])
        S.sort()
        dfs(0, 0, [])
        return res


# 78. Path Sum II 
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @param sum, an integer
    # @return a list of lists of integers
    def pathSum(self, root, sum):
        res = []
        def dfs(root, target, valuelist):
            if root.left is None and root.right is None:
                if target == root.val:
                    res.append(valuelist + [root.val])
                    return
            if root.left is not None:
                dfs(root.left, target - root.val, valuelist + [root.val])
            if root.right is not None:
                dfs(root.right, target - root.val, valuelist + [root.val])
        
        if root is None:
            return []
        
        dfs(root, sum, [])
        return res


# 77. 3Sum Closest
class Solution:
    # @return an integer
    def threeSumClosest(self, num, target):
        num.sort()
        mindiff = 100000
        res = 0
        for i in range(len(num)):
            left = i + 1;
            right = len(num) - 1
            while left < right:
                sum = num[i] + num[left] + num[right]
                diff = abs(sum - target)
                if diff < mindiff: mindiff = diff; res = sum
                if sum == target:
                    return sum
                elif sum < target:
                    left += 1
                else:
                    right -= 1
        return res
    

# 76. Jump Game
class Solution:
    # @param A, a list of integers
    # @return a boolean
    def canJump(self, A):
        lenA = len(A)
        canReach = 0
        for i in xrange(lenA):
            if i <= canReach:
                canReach = max(canReach, i + A[i])
                if canReach >= lenA - 1:
                    return True
        return False


# 75. Longest Common Prefix 
class Solution:
    # @return a string
    def longestCommonPrefix(self, strs):
        if len(strs) == 0:
            return ''
        
        prefixlen = 0
        while True:
            prefix = strs[0][:prefixlen]
            for i in xrange(len(strs)):
                if prefixlen > len(strs[i]) or prefix != strs[i][:prefixlen]:
                    return strs[0][:prefixlen-1]
            prefixlen += 1


# 74. Unique Binary Search Tree ii
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @return a list of tree node
    def generateTrees(self, n):
        res = self.dfs(1, n)
        return res

    def dfs(self, start, end):
        if start > end:
            return [None]
        res = []
        for val in xrange(start, end + 1):
            LeftTree = self.dfs(start, val - 1)
            RightTree = self.dfs(val + 1, end)
            for i in LeftTree:
                for j in RightTree:
                    root = TreeNode(val)
                    root.left = i
                    root.right = j
                    res.append(root)
        return res


# 73. Count and Say
class Solution:
    # @return a string
    def countAndSay(self, n):
        if n == 0: return ''

        s = '1'
        for i in range(1, n):
            s = self.count(s)

        return s

    def count(self, s):
        res = ''
        i = 0
        lenS = len(s)
        count = 1
        while i < lenS:
            while i + 1 < lenS and s[i+1] == s[i]:
                count += 1
                i += 1
            res += str(count) + s[i]
            count = 1
            i += 1

        return res


# 72. 
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a list node
    # @return a tree node
    def sortedListToBST(self, head):
        if head is None: return None
        ptr = head
        values = []
        while ptr is not None:
            values.append(ptr.val)
            ptr = ptr.next
        return self.dfs(values)

    def dfs(self, values):
        if len(values) == 0: return None
        if len(values) == 1:
            return TreeNode(values[0])
        mid = len(values) / 2
        root = TreeNode(values[mid])
        root.left = self.dfs(values[:mid])
        root.right = self.dfs(values[mid + 1:])
        return root


# 71. Search for a Range
class Solution:
    # @param A, a list of integers
    # @param target, an integer to be searched
    # @return a list of length 2, [index1, index2]
    def searchRange(self, A, target):
        start = 0
        end = len(A) - 1
        mid = (start + end) / 2
        while -1 < start <= end < len(A):
            mid = (start + end) / 2
            if A[mid] > target:
                end = mid - 1
            elif A[mid] < target:
                start = mid + 1
            else:
                break

        if A[mid] != target:
            return [-1, -1]

        toleft = 1; toright = 1
        while mid - toleft > -1 and A[mid - toleft] == target:
            toleft += 1
        while mid + toright < len(A) and A[mid + toright] == target:
            toright += 1

        return [mid - toleft + 1, mid + toright - 1]


# 70. Subsets
class Solution:
    # @param S, a list of integer
    # @return a list of lists of integer
    def subsets(self, S):
        def dfs(valuelist, depth, start):
            res.append(valuelist)
            if depth == len(S): return
            for i in xrange(start, len(S)):
                dfs(valuelist + [S[i]], depth + 1, i + 1)

        S.sort()
        res = []
        dfs([], 0, 0)
        return res

# 69. Unique Paths II
class Solution:
    # @param obstacleGrid, a list of lists of integers
    # @return an integer
    def uniquePathsWithObstacles(self, obstacleGrid):
        if len(obstacleGrid) == 0:
            return 0

        dp = [[0 for i in xrange(len(obstacleGrid[0]))] for i in xrange(len(obstacleGrid))]

        # init
        if obstacleGrid[0][0] == 1:
            return 0
        dp[0][0] = 1

        for j in xrange(1, len(obstacleGrid[0])):
            if obstacleGrid[0][j] != 1:
                dp[0][j] = dp[0][j - 1]
            else:
                dp[0][j] = 0
        
        for i in xrange(1, len(obstacleGrid)):
            if obstacleGrid[i][0] != 1:
                dp[i][0] = dp[i - 1][0]
            else:
                dp[i][0] = 0
        
        for i in xrange(1, len(obstacleGrid)):
            for j in xrange(1, len(obstacleGrid[0])):
                if obstacleGrid[i][j] != 1:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
                else:
                    dp[i][j] = 0
        
        return dp[-1][-1]


# 68. Valid Sudoku
class Solution:
    # @param board, a 9x9 2D array
    # @return a boolean
    def isValidSudoku(self, board):
        for row in xrange(len(board)):
            tmp = []
            for col in xrange(len(board[0])):
                if board[row][col] != '.':
                    if board[row][col] not in tmp:
                        tmp.append(board[row][col])
                    else:
                        return False

        for col in xrange(len(board[0])):
            tmp = []
            for row in xrange(len(board)):
                if board[row][col] != '.':
                    if board[row][col] not in tmp:
                        tmp.append(board[row][col])
                    else:
                        return False

        for x in [0, 3, 6]:
            for y in [0, 3, 6]:
                tmp = []
                for row in [0,1,2]:
                    for col in [0,1,2]:
                        if board[x + row][y + col] != '.':
                            if board[x + row][y + col] not in tmp:
                                tmp.append(board[x + row][y + col])
                            else:
                                return False
        return True


# 67. Longest Consecutive Sequence
class Solution:
    # @param num, a list of integer
    # @return an integer
    def longestConsecutive(self, num):
        dict = {}
        for i in num:
            dict[i] = False
        longest = 0
        for key in dict.iterkeys():
            tmp = 1
            if dict[key] is False:
                dict[key] = True
                key1 = key
                key2 = key
                while dict.has_key(key1 + 1) and dict[key1 + 1] is False:
                    dict[key1 + 1] = True
                    tmp += 1
                    key1 += 1
                while dict.has_key(key2 - 1) and dict[key2 - 1] is False:
                    dict[key2 - 1] = True
                    tmp += 1
                    key2 -= 1
                longest = max(tmp, longest)
        return longest


# 66. Flatten Binary Tree to Linked List
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return nothing, do it in place
    def flatten(self, root):
        if root is None:
            return root
        if root.right is not None:
            self.flatten(root.right)
        if root.left is not None:
            self.flatten(root.left)
            ptr = root.left
            while ptr.right is not None:
                ptr = ptr.right
            ptr.right = root.right
            root.right = root.left
            root.left = None


# 65. Gray Code
i ^ (i>>1)

# 64. Trapping Rain Water
class Solution:
    # @param A, a list of integers
    # @return an integer

    def trap(self, A):
        if len(A) <= 2:
            return 0

        preHighest = [0] * len(A)
        preHighest[0] = A[0]
        subHighest = [0] * len(A)
        subHighest[-1] = A[-1]

        for i in xrange(1, len(A)):
            preHighest[i] = max(preHighest[i - 1], A[i])
        for j in xrange(len(A) - 2, -1, -1):
            subHighest[j] = max(subHighest[j + 1], A[j])

        trapwater = 0
        for k in xrange(1, len(A) - 1):
            if min(preHighest[k], subHighest[k]) - A[k] > 0:
                trapwater += min(preHighest[k], subHighest[k]) - A[k]

        return trapwater


# 63. Palindrome Number
class Solution:
    # @return a boolean

    def isPalindrome(self, x):
        if x < 0:
            return False
        k = 1
        while x // k >= 10:
            k *= 10

        while x > 0:
            if x // k != x % 10:
                return False
            x = (x - x // k * k) // 10
            k //= 100

        return True


# 62. Length of Last Word
class Solution:
    # @param s, a string
    # @return an integer

    def lengthOfLastWord(self, s):
        # last index
        tail = len(s) - 1
        # get index of the last char which is not space
        while tail > -1:
            if s[tail] == ' ':
                tail -= 1
            else:
                break
        if tail == -1:
            return 0

        head = tail - 1
        while head >= 0:
            if s[head] != ' ':
                head -= 1
            else:
                break

        return tail - head


# 61. Minimum Depth of Binary Tree
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return an integer

    def minDepth(self, root):
        if root == None:
            return 0
        if root.left == None and root.right != None:
            return self.minDepth(root.right) + 1
        elif root.right == None and root.left != None:
            return self.minDepth(root.left) + 1
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1


# 60. Sum Root to Leaf Numbers
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return an integer

    def sumNumbers(self, root):
        return self.sum(root, 0)

    def sum(self, root, preSum):
        if root == None:
            return 0
        elif root.left == None and root.right == None:
            return preSum + root.val
        else:
            return self.sum(root.left, (root.val + preSum) * 10) + self.sum(root.right, (root.val + preSum) * 10)

# 59. Remove Nth Node From End of List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    # @return a ListNode

    def removeNthFromEnd(self, head, n):
        if head == None:
            return None
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        s = dummy
        for i in xrange(n):
            s = s.next

        while s.next != None:
            s = s.next
            p = p.next

        p.next = p.next.next

        return dummy.next


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
                for j in self.dfs(k - 1, A[i + 1:]):
                    ret.append([A[i]] + j)
            return ret


# 57. Remove Duplicates from Sorted Array II
class Solution:
    # @param A a list of integers
    # @return an integer

    def removeDuplicates(self, A):
        if len(A) <= 2:
            return len(A)
        slow = 0
        count = 1
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
        len1 = len(word1)
        len2 = len(word2)
        if len(word1) == 0:
            return len2
        if len(word2) == 0:
            return len1

        dp = [[0 for j in xrange(len2 + 1)] for i in xrange(len1 + 1)]

        for i in xrange(1, len1 + 1):
            dp[i][0] = i
        for j in xrange(1, len2 + 1):
            dp[0][j] = j

        for i in xrange(1, len1 + 1):
            for j in xrange(1, len2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

        return dp[len1][len2]

#
# 55. Valid Parenthese


class Solution:
    # @return a boolean

    def isValid(self, s):
        right = ")]}"
        left = "([{"
        if len(s) == 0:
            return True
        if len(s) == 1:
            return False

        if s[0] in right:
            return False

        dict = {'(': ')', '[': ']', '{': '}'}
        stack = []
        stack.append(s[0])
        i = 1

        while i < len(s):
            if s[i] in right:
                if stack:
                    tmp = stack.pop()
                    if dict[tmp] != s[i]:
                        return False
                else:
                    return False
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
