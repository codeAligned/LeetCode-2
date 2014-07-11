__author__ = 'le0nh@rdt'

# Binary Tree Level Order Traversal
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
        res = []
        self.preorder(root, 0, res)
        return res

    def preorder(self, root, level, res):
        if root:
            if len(res) < level + 1: res.append([])
            res[level].append(root.val)
            self.preorder(root.left, level+1, res)
            self.preorder(root.right, level+1, res)





# 3Sum Closest
class Solution:
    # @return an integer
    def threeSumClosest(self, num, target):
        num.sort()
        mindiff=100000
        res=0
        for i in range(len(num)):
            left=i+1; right=len(num)-1
            while left<right:
                sum=num[i]+num[left]+num[right]
                diff=abs(sum-target)
                if diff<mindiff: mindiff=diff; res=sum
                if sum==target: return sum
                elif sum<target: left+=1
                else: right-=1
        return res




# Binary Tree Zigzag Level Order Traversal
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a list of lists of integers
    def zigzagLevelOrder(self, root):
        res = []
        self.preorder(root, 0, res)
        for i in xrange(len(res)):
            if i%2 == 1:
                res[i].reverse()
        return res


    def preorder(self, root, level, res):
        if root:
            if len(res) < level + 1: res.append([])
            res[level].append(root.val)
            self.preorder(root.left, level + 1, res)
            self.preorder(root.right, level + 1, res)



#Triangle
class Solution:

    # @param triangle, a list of lists of integers
    # @return an integer
    def minimumTotal(self, triangle):
        for i in xrange(1, len(triangle)):
            for j in xrange(len(triangle[i])):
                if j == 0:
                    triangle[i][0] += triangle[i - 1][0]
                elif j == len(triangle[i]) - 1:
                    triangle[i][-1] += triangle[i - 1][-1]
                else:
                    triangle[i][j] += min(triangle[i-1][j-1],triangle[i-1][j])

        return min(triangle[-1])



#Count and Say
class Solution:
    # @return a string
    def countAndSay(self, n):
        s='1'
        for i in xrange(2, n+1):
            s = self.count(s)
        return s
    def count(self, s):
        t = ''
        count = 0
        curr = '#'
        for i in s:
            if i!= curr:
                if curr != '#':
                    t+=str(count)+curr
                curr=i
                count = 1
            else:
                count += 1
        t+=str(count)+curr
        return t

#Subset II
class Solution:
    # @param num, a list of integer
    # @return a list of lists of integer
    def subsetsWithDup(self, S):
        def dfs(depth, start, valuelist):
            if valuelist not in res: res.append(valuelist)
            if depth == len(S): return
            for i in xrange(start, len(S)):
                dfs(depth+1, i+1, valuelist+[S[i]])
        S.sort()
        res = []
        dfs(0, 0, [])
        return res

# Merge Two Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param two ListNodes
    # @return a ListNode
    def mergeTwoLists(self, l1, l2):
         if l1 == None: return l2
         if l2 == None: return l1

         if l1.val <= l2.val:
             node = l1
             head = l1
             l1 = l1.next
         else:
             node = l2
             head = l2
             l2 = l2.next
         while l1 != None and l2 != None:
              if l1.val <= l2.val:
                   node.next = l1
                   node = node.next
                   l1 = l1.next
              else:
                   node.next = l2
                   node = node.next
                   l2 = l2.next

         if l1 != None: node.next = l1
         elif l2 != None: node.next = l2
         return head

#Balanced Binary Tree
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
         return self.check(root)[1]

    def check(self, root):
         if root == None: return (0, True)
         LH, LB = self.check(root.left)
         RH, RB = self.check(root.right)
         return (max(LH, RH)+1, LB and RB and abs(LH-RH) <= 1)

#Symmetric Tree
# (1) 递归法
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

         return self.check(root.left, root.right)

    def check(self, p, q):
         if p == None and q == None: return True
         elif p == None or q == None: return False

         return p.val == q.val and self.check(p.left,q.right) and self.check(p.right, q.left)

#(2)非递归法
#Remove Duplicates from Sorted Array
class Solution:
    # @param a list of integers
    # @return an integer
    def removeDuplicates(self, A):
        if len(A) <= 1:
            return len(A)
        i = 0
        j = 1
        while j < len(A):
            if A[i] == A[j]:
                j+=1
            else:
                i+=1
                A[i] = A[j]
                j+=1
        return i+1

class Solution:
    # @param num, a list of integer
    # @return a list of lists of integers
    def permute(self, num):
        if len(num) == 0: return []
        if len(num) == 1: return [num]
        res = []
        for i in xrange(len(num)):
            for j in self.permute(num[0:i]+num[i+1:]):
                res.append(j + [num[i]])
        return res


#Generate Parentheses
class Solution:
    # @param an integer
    # @return a list of string
    def generateParenthesis(self, n):
        return self.helper(n, n)
    def helper(self, n, m):
        if n == 0: return [")"*m]
        elif n == m: return ["(" + i for i in self.helper(n-1, m)]
        else: return ["(" + i for i in self.helper(n-1,m)] + [")" + i for i in self.helper(n,m-1)]



#Search in Rotated Sorted Array
class Solution:
    # @param A, a list of integers
    # @param target, an integer to be searched
    # @return an integer
    def search(self, A, target):
        l = 0
        r = len(A) - 1
        while l <= r:
            m = (l+r) // 2
            if A[m] == target:
                return m
            elif A[m] >= A[l]:
                if A[l] <= target and A[m] >= target:
                    r = m - 1
                else: l = m + 1
            else:
                if A[m] <= target and A[r] >= target:
                    l = m + 1
                else: r = m - 1
        return -1



#Palindrome Number
class Solution:
    # @return a boolean
    def isPalindrome(self, x):
        if x < 0:
            return False
        x_string = str(x)
        x_reverse = self.reverseStr(x_string)
        if x_string == x_reverse:
            return True
        else: return False
    def reverseStr(self, x_str):
        if len(x_str) <= 1:
            return x_str
        return self.reverseStr(x_str[1:])+x_str[0]



#Sum Root to Leaf Numbers
#Timeout code:
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
        sum = 0
        stack1 = []
        stack2 = []
        if root == None:
            return 0
        stack1.append(root)
        if root.left != None:
            stack1.append(root.left)
        if root.right != None:
            stack1.append(root.right)
        stack2.append(stack1.pop())
        while len(stack1) > 0:
            top = stack1[-1]
            if top.left != None:
                stack1.append(top.left)
            if top.right != None:
                stack1.append(top.right)
            if top.left == None and top.right == None:
                sum += self.helper(stack1)
        return sum

    def helper(self,stack):
        s = []
        for element in stack:
            s.append(element.val)
        ss = ''.join(map(str, s))
        return int(ss)

# Correction:
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
   def sum(self,root,preSum):
        if root == None: return 0
        preSum = root.val + 10 * preSum
        if root.left == None and root.right == None: return preSum
        return self.sum(root.left, preSum) + self.sum(root.right, preSum)



# Combinations
class Solution:
    # @return a list of lists of integers
    def combine(self, n, k):
        return self.getComb([], n, 0, k)
    def getComb(self, res, n, pos, k):
        flag = pos + 1
        _res = []
        if k == 1:
            for i in range(pos+1, n+1):
                _res.append([i])
        elif k > 1:
            while pos + k <= n:
                for x in self.getComb(res, n, pos+1, k-1):
                    _res.append(x)
                    _res[-1].insert(0, pos+1)
                pos += 1
        return _res



# Subsets
class Solution:
    # @param S, a list of integer
    # @return a list of lists of integer
    def subsets(self, S):
        S.sort()
        result = [[]]
        for x in range(0,len(S)+1):
            temp = self.combine(S, x)
            for elemt in temp:
                result.append(elemt)
        return result
    def combine(self, S, k):
        res = self.getComb([], S, 0, k)
        return res
    def getComb(self, res, S, pos, k):
        flag = pos + 1
        _res = []
        if k == 1:
            for i in range(pos, len(S)):
                _res.append([S[i]])
        elif k > 1:
            while pos + k <= len(S):
                for x in self.getComb(res, S, pos+1, k-1):
                    _res.append(x)
                    _res[-1].insert(0, S[pos])
                pos += 1
        return _res



# Minimum Depth of Binary Tree
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
        if root == None: return 0
        elif root.left == None and root.right == None: return 1
        elif root.left == None and root.right != None:
            return self.minDepth(root.right) + 1
        elif root.right == None and root.left != None:
            return self.minDepth(root.left) + 1
        elif root.left != None and root.right != None:
            return min(self.minDepth(root.left), self.minDepth(root.right))+1



# Length of Last Word
class Solution:
    # @param s, a string
    # @return an integer
    def lengthOfLastWord(self, s):
        if len(s) == 0: return 0
        elif s[-1] == " ":
            s = s[:-1]
            return self.lengthOfLastWord(s)
        else:
            words = s.split(" ")
            word = words[-1]
            return len(word)



#Sudoku
class Solution:
    # @param board, a 9x9 2D array
    # @return a boolean
    def isValidSudoku(self, board):
        valid = False
        for i in range(0, 3):
            for j in range(0, 3):
                smallboard = []
                for ii in range(0, 3):
                    for jj in range(0, 3):
                        smallboard.append(board[i * 3 + ii][j * 3 + jj])
                valid = valid and self.isValidSudokuSmall(smallboard)
        valid = valid and self.checkLines(board)
        return valid
    def isValidSudokuSmall(self, smallboard):  # small board is a 3*3 board and all num of it should be unique.
        smallboard = smallboard.remove('.')
        if len(smallboard) == len(set(smallboard)):
            return True
        return False
    def checkLines(self, board):
        for line in range(0,10):
            lset = board[line].remove('.')
            if len(lset) != len(set(lset)):
                return False
        for row in range(0,10):
            rset = board[:, row].remove('.')
            if len(rset) != len(set(rset)):
                return False
        return True




# Trapping Rain Water

class Solution:
    # @param A, a list of integers
    # @return an integer
    def trap(self, A):
        if len(A) <= 2: return 0
        LR = self.fromLtoR(A)
        RL = self.fromRtoL(A)
        sum = 0
        for i in range(0, len(A)):
            tmp = min(LR[i],RL[i])-A[i]
            if tmp > 0:
                sum += tmp
        return sum
    def fromLtoR(self, A):
        max = A[0]
        LR = [0]*len(A)
        for i in range(1, len(A)):
            if A[i] > max:
                max = A[i]
                LR[i] = max
            else:
                LR[i] = max
        return LR
    def fromRtoL(self, A):
        max = A[-1]
        RL = [0]*len(A)
        for i in range(len(A)-2, -1, -1):
            if A[i] > max:
                max = A[i]
                RL[i] = max
            else:
                RL[i] = max
        return RL




# Search in Rotated Sorted Array II
class Solution:
    # @param A, a list of integers
    # @param target, an integer to be searched
    # @return an integer
    def search(self, A, target):
        l = 0
        r = len(A)-1
        while l <= r:
            m = (r+l) // 2
            if A[m] == target: return True
            elif A[m] > A[l]:
                if target <= A[m] and target >= A[l]:
                    r = m - 1
                else:
                    l = m + 1
            elif A[m] < A[l]:
                if target >= A[m] and target <= A[r]:
                    l = m + 1
                else:
                    r = m - 1
            elif A[m] == A[l]:
                l += 1
        return False




# Populating Next Right Pointers in Each Node
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
        head = root
        if head == None: return
        if root.left != None:
            root.left.next = root.right
        if root.right != None:
            if root.next != None:
                root.right.next = root.next.left
            else:
                root.right.next = None
        self.connect(root.left)
        self.connect(root.right)




# Unique Paths
class Solution:
    # @return an integer
    def uniquePaths(self, m, n):
        arr = []
        if m == 1 or n == 1: return 1
        for i in xrange(m):
            arr.append([])
            for j in xrange(n):
                arr[i].append(0)
        arr[0][0] = 0
        for i in xrange(1,m):
            arr[i][0] = 1
        for j in xrange(1,n):
            arr[0][j] = 1
        for i in xrange(1, m):
            for j in xrange(1,n):
                arr[i][j] = arr[i][j-1] + arr[i-1][j]
        return arr[-1][-1]



# Unique Paths II
class Solution:
    # @param obstacleGrid, a list of lists of integers
    # @return an integer
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        arr = [[0 for j in xrange(n)] for i in xrange(m)]
        for j in xrange(0,n):
            if obstacleGrid[0][j] == 0:
                arr[0][j] = 1
            else: break
        for i in xrange(0, m):
            if obstacleGrid[i][0] == 0:
                arr[i][0] = 1
            else: break
        for i in xrange(1, m):
            for j in xrange(1,n):
                if obstacleGrid[i][j] == 1:
                    arr[i][j] = 0
                else:
                    arr[i][j] = arr[i][j-1] + arr[i-1][j]
        return arr[m-1][n-1]




# Valid Sudoku
class Solution:
    # @param board, a 9x9 2D array
    # @return a boolean
    def isValidSudoku(self, board):
        return self.checkRow(board) and self.checkLine(board) and self.checkBox(board)
    def checkRow(self,board):
        for row in board:
            dump = []
            for i in row:
                if i != '.':
                    if i not in dump:
                        dump.append(i)
                    else: return False
        return True
    def checkLine(self,board):
        for j in xrange(9):
            dump = []
            for i in xrange(9):
                k = board[i][j]
                if k != '.':
                    if k not in dump:
                        dump.append(k)
                    else:return False
        return True
    def checkBox(self,board):
        for i in xrange(3):
            for j in xrange(3):
                dump = []
                for m in xrange(3):
                    for n in xrange(3):
                        x = board[i*3+m][j*3+n]
                        if x != '.':
                            if x not in dump:
                                dump.append(x)
                            else:return False
        return True




# 解法1：（超时了）
class Solution:
    # @param num, a list of integer
    # @return an integer
    def longestConsecutive(self, num):
        maxConsecutive = 1
        for i in num:
            MCA, num = self.getMaxConsecutiveAscending(i, num)
            MCD, num = self.getMaxConsecutiveDecending(i, num)
            if MCA + MCD + 1 > maxConsecutive:
                maxConsecutive = MCA + MCD + 1
        return maxConsecutive
    def getMaxConsecutiveAscending(self, i, num):
        MCA = 0
        while i + 1 in num:
            MCA += 1
            num.remove(i+1)
            i += 1
        return MCA, num
    def getMaxConsecutiveDecending(self, i, num):
        MCD = 0
        while i - 1 in num:
            MCD += 1
            num.remove(i-1)
            i -= 1
        return MCD, num
# 解法二：
class Solution:
    # @param num, a list of integer
    # @return an integer
    def longestConsecutive(self, num):
        dict = {x:False for x in num}
        maxLen = -1
        for i in dict:
            if dict[i] == False:
                curr = i+1; lenAscending = 0
                while curr in dict and dict[curr] == False:
                    lenAscending += 1; dict[curr] = True; curr += 1
                curr = i - 1; lenDecending = 0
                while curr in dict and dict[curr] == False:
                    lenDecending += 1; dict[curr] = True; curr -= 1
                maxLen = max(maxLen, lenAscending + lenDecending + 1)
        return maxLen




# Path Sum II
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
        if root == None: return []
        Solution.sum = sum
        Solution.valArray = []
        self.getPath(root, [root.val], root.val)
        return Solution.valArray
    def getPath(self, root, pathArray, currSum):
        if root.left == None and root.right == None:
            if currSum == Solution.sum:
                Solution.valArray.append(pathArray)
                return
        if root.left != None:
            self.getPath(root.left, pathArray+[root.left.val], currSum + root.left.val)
        if root.right != None:
            self.getPath(root.right, pathArray+[root.right.val], currSum + root.right.val)




# Flatten Binary Tree to Linked List

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
        if root == None:return
        self.flatten(root.left)
        self.flatten(root.right)
        if root.left == None:
            return
        else:
            p = root.left
            while p.right != None: p = p.right
            p.right = root.right
            root.right = root.left
            root.left = None



# Jump Game
# 解法1：（超时）
class Solution:
    # @param A, a list of integers
    # @return a boolean
    def canJump(self, A):
        if len(A) <= 1: return True
        B = [0]* len(A)
        for i in xrange(len(A)):
            for j in xrange(min(A[i]+1,len(A) - i)):
                B[i+j] = max(A[i]-j, A[i+j])
        for x in B[:-1]:
            if x == 0: return False
        return True

# 解法2：(超时)
class Solution:
    # @param A, a list of integers
    # @return a boolean
    def canJump(self, A):
        if len(A) <= 1: return True
        B = [0]* len(A)
        i = 0
        while i < len(A):
            j = 0
            while j < min(A[i],len(A) - i):
                if A[i+j] > A[i] - j:
                    B[i+j] = A[i+j]
                    i = i + j - 1
                    break
                else:
                    B[i+j] = A[i] - j
                    j += 1
            i += 1
        for x in B[:-1]:
            if x == 0: return False
        return True

# 解法3：
class Solution:
    # @param A, a list of integers
    # @return a boolean
    def canJump(self, A):
        lenA = len(A)
        canReach = 0
        for i in xrange(lenA):
            if i <= canReach:
                canReach = max(canReach, i + A[i])
                if canReach >= lenA - 1: return True
        return False




# Longest Common Prefix
class Solution:
    # @return a string
    def longestCommonPrefix(self, strs):
        if len(strs) == 0: return ""
        longest = len(strs[0])
        for string in strs[1:]:
            index = 0
            while index < len(string) and index < longest and strs[0][index] == string[index]:
                index += 1
            longest = min(longest, index)
        return strs[0][:longest]




# Search for a Range
class Solution:
    # @param A, a list of integers
    # @param target, an integer to be searched
    # @return a list of length 2, [index1, index2]
    def searchRange(self, A, target):
        if len(A) <= 0: return [-1, -1]
        center = self.locate(A, target)
        if center == -1: return [-1, -1]
        counter_d = 0; counter_a = 0
        while center - counter_d >= 0:
            if A[center - counter_d] == target: counter_d += 1
            else: break
        while counter_a + center < len(A):
            if A[counter_a + center] == target: counter_a += 1
            else: break
        return [center - counter_d + 1, center + counter_a - 1]
    def locate(self, A, target):
        l = 0
        s = len(A)-1
        while l <= s:
            m = (l+s) // 2
            if target > A[m]:
                l = m + 1
            elif target < A[m]:
                s = m - 1
            else: return m
        return -1




# Convert Sorted List to Binary Search Tree
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
        num = []; curr = head
        while curr != None:
            num.append(curr.val)
            curr = curr.next
        return self.treeGenerator(num, 0, len(num) - 1)
    def treeGenerator(self, num, start, end):
        if start > end: return None
        mid = (start + end) // 2
        root = TreeNode(num[mid])
        root.left = self.treeGenerator(num, start, mid-1)
        root.right = self.treeGenerator(num, mid + 1, end)
        return root




#N-Queens II
class Solution:
    # @return an integer
    def totalNQueens(self, n):
        self.res = 0
        self.solve(n, 0, [-1 for i in xrange(n)])
        return self.res
    def solve(self, n, currQueenNum, board):
        if currQueenNum == n: self.res += 1; return
        for i in xrange(n):
            valid = True
            for k in xrange(currQueenNum):
                if board[k] == i: valid = False; break
                if abs(board[k] - i) == currQueenNum - k: valid = False; break
            if valid:
                board[currQueenNum] = i
                self.solve(n, currQueenNum+1, board)
