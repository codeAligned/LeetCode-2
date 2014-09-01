__author__ = 'le0nh@rdt'
# Permutation Sequence
# 2. 
class Solution:
    # @return a string
    def getPermutation(self, n, k):
        s = ''
        k -= 1
        blocksize = math.factorial(n - 1)
        factorial = blocksize * n
        num = [i for i in xrange(1, n + 1)]
        res = ''
        for i in reversed(xrange(1, n + 1)):
            res += str(num[k / blocksize])
            num.remove(num[k / blocksize])
            if i != 1:
                k = k % blocksize
                blocksize /= i - 1

        return res
        
# 1. 超时
class Solution:
    # @return a string
    def getPermutation(self, n, k):
        s = ''
        for i in xrange(1, n + 1):
            s += str(i)

        return self.recPerm(n, s, k - 1)

    def recPerm(self, n, s, k):
        if n == 1: return s
        blocksize = math.factorial(n - 1)
        tot = blocksize * n
        numOfBlock = k / blocksize
        return s[numOfBlock] + self.recPerm(n - 1, s[:numOfBlock] + s[numOfBlock + 1:], k % blocksize)


# Merge Intervals
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    # @param intervals, a list of Interval
    # @return a list of Interval
    def merge(self, intervals):
        if len(intervals) == 0: return []
        if len(intervals) == 1: return intervals
        intervals.sort(key=lambda x:x.start)
        res = []
        tmp = intervals[0]
        for i in xrange(1, len(intervals)):
            if intervals[i].start <= tmp.end:
                tmp.start = min(tmp.start, intervals[i].start)
                tmp.end = max(tmp.end, intervals[i].end)
            elif intervals[i].start > tmp.end:
                res.append(tmp)
                tmp = intervals[i]
        
        res.append(tmp)
        return res


# Sort List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # @param head, a ListNode
    # @return a ListNode
    def sortList(self, head):
        if head is None: return head
        if head.next is None: return head

        slow = head
        fast = head

        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next

        head1 = head
        head2 = slow.next
        slow.next = None

        head1 = self.sortList(head1)
        head2 = self.sortList(head2)

        return self.merge(head1, head2)

    def merge(self, head1, head2):
        if head1 is None: return head2
        elif head2 is None: return head1

        dummy = ListNode(0)
        p = dummy
        while head1 is not None and head2 is not None:
            if head1.val <= head2.val:
                p.next = head1
                head1 = head1.next
                p = p.next
            elif head1.val > head2.val:
                p.next = head2
                head2 = head2.next
                p = p.next

        if head2 is None and head1 is not None:
            p.next = head1
        elif head2 is not None and head1 is None:
            p.next = head2

        return dummy.next


# Restore IP Address
class Solution:
    # @param s, a string
    # @return a list of strings
    def restoreIpAddresses(self, s):
        def dfs(s, blocks, ips, ip):
            if blocks == 4:
                if s == '':
                    ips.append(ip[1:])
                return
            for i in xrange(1, 4):
                if i <= len(s):
                    if int(s[:i]) <= 255:
                        dfs(s[i:], blocks + 1, ips, ip + '.' + s[:i])
                    if s[0] == '0':
                        break

        ips = []
        dfs(s, 0, ips, '')
        return ips


# Insert Interval
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e
# Solution 2: 
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution: 
# treat newInterval as some cache, merge interval afterward to it. Then decide what to do next
    # @param intervals, a list of Intervals
    # @param newInterval, a Interval
    # @return a list of Interval
    def insert(self, intervals, newInterval):
        inserted = False
        res = []
        for i in xrange(len(intervals)):
            if intervals[i].end < newInterval.start:
                res.append(intervals[i])
            elif intervals[i].start > newInterval.end:
                if not inserted:
                    inserted = True
                    res.append(newInterval)
                res.append(intervals[i])
            else:
                newInterval.start = min(newInterval.start, intervals[i].start)
                newInterval.end = max(newInterval.end, intervals[i].end)

        if len(res) == 0 or newInterval.start > res[-1].end:
            res.append(newInterval)

        return res

# Solution 1: time limit exceeded
class Solution:
    # @param intervals, a list of Intervals
    # @param newInterval, a Interval
    # @return a list of Interval
    def insert(self, intervals, newInterval):
        res = []
        if len(intervals) == 0: res.append(newInterval); return res
        if len(intervals) == 1:
            if self.checkOverlap(intervals[0], newInterval):
                res.append(Interval(min(intervals[0].start, newInterval.start), max(intervals[0].end, newInterval.end)))
            else:
                if newInterval.start > intervals[0].end:
                    res.append(intervals[0]);res.append(newInterval)
                else:
                    res.append(newInterval); res.append(intervals[0])
            return res
        alreadyInserted = False
        noOverlapping = False
        res.append(intervals[0])
        i = 1
        while i < len(intervals):
            if alreadyInserted:
                if not noOverlapping:
                    if res[-1].end < intervals[i].start:
                        noOverlapping = True
                        res.append(intervals[i])
                        i += 1
                    else:
                        res[-1].end = max(res[-1].end, intervals[i].end)
                else:
                    res.append(intervals[i])
                    i+=1
            else:
                if self.checkOverlap(res[-1], newInterval):
                    res[-1].start = min(res[-1].start, newInterval.start)
                    res[-1].end = max(res[-1].end, newInterval.end)
                    alreadyInserted = True
                else:
                    res.append(intervals[i])
                    i += 1
        return res


    def checkOverlap(self, a, b):
        if a.start <= b.end <= a.end or \
                                    a.start <= b.start <= a.end or \
                                    b.start <= a.start <= b.end or \
                                    b.start <= a.end <= b.end:
            return True
        else: return False


# Word Breaker
class Solution:
    # @param s, a string
    # @param dict, a set of string
    # @return a boolean
    def wordBreak(self, s, dict):
        db = [True]
        for i in xrange(0, len(s)):
            db.append(False)
            for j in xrange(i, -1, -1):
                if db[j] and s[j: i+1] in dict:
                    db[i+1] = True
                    break
        return db[-1]


# 4Sum
# 直接用set要比用list再去重的速度要快一点点（十几毫秒）


class Solution:
    # @return a list of lists of length 4, [[val1,val2,val3,val4]]

    def fourSum(self, num, target):
        numLen = len(num)
        d = {}
        res = set()
        num.sort()
        if numLen < 4:
            return []

        for i in xrange(numLen):
            for j in xrange(i + 1, numLen):
                if (num[i] + num[j]) not in d:
                    d[(num[i] + num[j])] = [(i, j)]
                else:
                    d[(num[i] + num[j])].append((i, j))

        for m in xrange(numLen):
            for n in xrange(m + 1, numLen - 2):
                key = target - num[m] - num[n]
                if key in d:
                    for item in d[key]:
                        if item[0] > n:
                            res.add(
                                (num[m], num[n], num[item[0]], num[item[1]]))

        return [list(i) for i in res]


# Sudoku Solver (using mask and 4 times faster than previous one)
class Solution:
    # @param board, a 9x9 2D array
    # Solve the Sudoku by modifying the input board in-place.
    # Do not return any value.

    def solveSudoku(self, board):
        rmask, cmask, bmask = [0] * 9, [0] * 9, [0] * 9
        for i in range(9):
            for j in range(9):
                b = i / 3 * 3 + j / 3
                if board[i][j] != '.':
                    change = 1 << (int(board[i][j]) - 1)
                    self.modifyMask(rmask, cmask, bmask, i, j, b, change)
        self.dfs(board, 0, rmask, cmask, bmask)

    def modifyMask(self, rmask, cmask, bmask, i, j, b, change):
        rmask[i] ^= change
        cmask[j] ^= change
        bmask[b] ^= change

    def dfs(self, board, k, rmask, cmask, bmask):
        if k == 81:
            return True
        i, j = k / 9, k % 9
        b = i / 3 * 3 + j / 3
        if board[i][j] != '.':
            return self.dfs(board, k + 1, rmask, cmask, bmask)
        for digit in range(9):
            change = 1 << digit
            if rmask[i] & change == 0 and cmask[j] & change == 0 and bmask[b] & change == 0:
                self.modifyMask(rmask, cmask, bmask, i, j, b, change)
                board[i][j] = str(digit + 1)
                if self.dfs(board, k + 1, rmask, cmask, bmask):
                    return True
                board[i][j] = '.'
                self.modifyMask(rmask, cmask, bmask, i, j, b, change)
        return False


# Sudoku Solver (Time exceed but works)
class Solution:
    # @param board, a 9x9 2D array
    # Solve the Sudoku by modifying the input board in-place.
    # Do not return any value.

    def solveSudoku(self, board):
        def dfs(board):
            for i in xrange(9):
                for j in xrange(9):
                    if board[i][j] == '.':
                        for k in "123456789":
                            board[i][j] = k
                            if isValid(i, j) and dfs(board):
                                return True
                            board[i][j] = '.'
                        return False
            return True

        def isValid(x, y):
            tmp = board[x][y]
            board[x][y] = 'D'
            for i in xrange(9):
                if board[i][y] == tmp:
                    return False
            for j in xrange(9):
                if board[x][j] == tmp:
                    return False
            for i in xrange(3):
                for j in xrange(3):
                    if board[(x // 3) * 3 + i][(y // 3) * 3 + j] == tmp:
                        return False
            board[x][y] = tmp
            return True

        dfs(board)


# Maximal Rectangle
class Solution:
    # @param matrix, a list of lists of 1 length string
    # @return an integer

    def maximalRectangle(self, matrix):
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[0])):
                if matrix[i][j] == '1':
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
        for i in xrange(1, len(matrix)):
            for j in xrange(len(matrix[0])):
                if matrix[i][j] != 0:
                    matrix[i][j] += matrix[i - 1][j]
        maximalRec = 0
        for height in matrix:
            maximalRec = max(maximalRec, self.largestRectangleArea(height))
        return maximalRec

    def largestRectangleArea(self, height):
        height_stack = []
        index_stack = []
        largestRec = 0
        for i in xrange(len(height)):
            if (not height_stack) or height[i] > height_stack[-1]:
                height_stack.append(height[i])
                index_stack.append(i)
            elif height[i] < height_stack[-1]:
                lastIndex = 0
                while height_stack and height_stack[-1] > height[i]:
                    lastIndex = index_stack.pop()
                    width = i - lastIndex
                    s = width * height_stack.pop()
                    largestRec = max(largestRec, s)
                height_stack.append(height[i])
                index_stack.append(lastIndex)
        while height_stack:
            largestRec = max(
                largestRec, height_stack.pop() * (len(height) - index_stack.pop()))

        return largestRec


# Largest Rectangle in Histogram
class Solution:
    # @param height, a list of integer
    # @return an integer

    def largestRectangleArea(self, height):
        height_stack = []
        index_stack = []
        largestRec = 0
        for i in xrange(len(height)):
            if (not height_stack) or height[i] > height_stack[-1]:
                height_stack.append(height[i])
                index_stack.append(i)
            elif height[i] < height_stack[-1]:
                lastIndex = 0
                while height_stack and height_stack[-1] > height[i]:
                    lastIndex = index_stack.pop()
                    width = i - lastIndex
                    s = width * height_stack.pop()
                    largestRec = max(largestRec, s)
                height_stack.append(height[i])
                index_stack.append(lastIndex)
        while height_stack:
            largestRec = max(
                largestRec, height_stack.pop() * (len(height) - index_stack.pop()))

        return largestRec


# Merge K Sorted List
# 2) Heap structure
class Solution:
    # @param a list of ListNode
    # @return a ListNode

    def mergeKLists(self, lists):
        heap = []
        for node in lists:
            if node != None:
                heap.append((node.val, node))
        heapq.heapify(heap)
        head = ListNode(0)
        curr = head
        while heap:
            pop = heapq.heappop(heap)
            curr.next = ListNode(pop[0])
            curr = curr.next
            if pop[1].next:
                heapq.heappush(heap, (pop[1].next.val, pop[1].next))
        return head.next

# 1) Time Exceed


class Solution:
    # @param a list of ListNode
    # @return a ListNode

    def mergeKLists(self, lists):
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]

        dummy1 = ListNode(0)
        start = 0
        while lists[start] == None and start < len(lists) - 1:
            start += 1

        dummy1.next = lists[start]

        for j in xrange(start + 1, len(lists)):
            self.mergeTwo(dummy1.next, lists[j])

        return dummy1.next

    def mergeTwo(self, l1, l2):
        if l1 == None:
            return l2
        if l2 == None:
            return l1

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

        if l1 != None:
            node.next = l1
        elif l2 != None:
            node.next = l2
        return head


# Rotate List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a ListNode
    # @param k, an integer
    # @return a ListNode

    def rotateRight(self, head, k):
        if head == None:
            return head

        dummy = ListNode(0)
        dummy.next = head
        s = dummy
        t = dummy
        i = 0

        while i < k:
            if t.next:
                t = t.next
            else:
                t = dummy.next
            i += 1

        while t.next:
            t = t.next
            s = s.next

        if s == dummy:
            return head
        else:
            t.next = dummy.next
            dummy.next = s.next
            s.next = None
            return dummy.next


# First Missing Positive
# hint: using bucket sort
class Solution:
    # @param A, a list of integers
    # @return an integer

    def firstMissingPositive(self, A):

        for i in xrange(len(A)):
            while A[i] != i + 1:
                if A[i] <= 0 or A[i] > len(A) or A[i] == A[A[i] - 1]:
                    break
                t = A[A[i] - 1]
                A[A[i] - 1] = A[i]
                A[i] = t

        for i in xrange(len(A)):
            if A[i] != i + 1:
                return i + 1

        return len(A) + 1


# Best Time to Buy and Sell Stock III
class Solution:
    # @param prices, a list of integer
    # @return an integer

    def maxProfit(self, prices):
        if len(prices) <= 1:
            return 0

        lowPrice = prices[0]
        maxProfitForward = []
        maxProfit = 0
        for price in prices:
            lowPrice = min(lowPrice, price)
            maxProfit = max(maxProfit, price - lowPrice)
            maxProfitForward.append(maxProfit)

        highPrice = prices[-1]
        maxProfitBackward = []
        maxProfit = 0
        for price in reversed(prices):
            highPrice = max(highPrice, price)
            maxProfit = max(maxProfit, highPrice - price)
            maxProfitBackward.append(maxProfit)
        maxProfitBackward.reverse()

        maxProfit = 0
        for i in xrange(len(prices)):
            maxProfit = max(
                maxProfit, maxProfitForward[i] + maxProfitBackward[i])
        return maxProfit


# Sqrt(x)
class Solution:
    # @param x, an integer
    # @return an integer

    def sqrt(self, x):
        if x == 0:
            return 0
        start = 0
        end = x
        while start < end:
            mid = (start + end) // 2
            if mid * mid > x:
                end = mid
            elif (mid + 1) * (mid + 1) < x:
                start = mid + 1
            elif mid * mid == x:
                return mid
            elif (mid + 1) * (mid + 1) == x:
                return mid + 1
            else:
                return mid


# Longest Substring Without Repeating Characters
class Solution:
    # @return an integer

    def lengthOfLongestSubstring(self, s):
        if len(s) <= 1:
            return len(s)
        start = 0
        end = 1
        maxLen = 1
        while end < len(s):
            if s[end] in s[start:end]:
                maxLen = max(end - start, maxLen)
                start = s[start:end].index(s[end]) + 1 + start

            end += 1
        maxLen = max(end - start, maxLen)
        return maxLen

# Anagrams


class Solution:
    # @param strs, a list of strings
    # @return a list of strings

    def anagrams(self, strs):
        if len(strs) <= 1:
            return []
        d = dict()

        for s in strs:
            key = ''.join(sorted(s))
            if d.has_key(key):
                d[key].append(s)
            else:
                d[key] = [s]

        ret = []
        for key in d:
            if len(d[key]) > 1:
                ret += d[key]
        return ret


# Scramble String
# Recursive
class Solution:
    # @return a boolean

    def isScramble(self, s1, s2):
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        l1 = list(s1)
        l2 = list(s2)
        l1.sort()
        l2.sort()
        if l1 != l2:
            return False
        length = len(s1)
        for i in xrange(1, length):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[i:], s2[:length - i]) and self.isScramble(s1[:i], s2[length - i:]):
                return True
        return False


# First Missing Positive (My solution)
# 编译错误，似乎是LeetCode不允许使用set method
class Solution:
    # @param A, a list of integers
    # @return an integer

    def firstMissingPositive(self, A):
        B = list(set(A))
        B.sort()
        i = 0
        while i < len(B) - 1:
            if B[i + 1] != B[i] + 1:
                if B[i + 1] < 1:
                    i += 1
                    continue
                elif B[i + 1] > 1:
                    if B[i] < 0:
                        return 1
                    elif B[i] >= 0:
                        return B[i] + 1
                elif B[i + 1] == 1:
                    return self.getFirstPos(i + 1, B)
            i += 1
        return B[-1] + 1 if B[-1] >= 0 else 1

    def getFirstPos(self, start, B):
        for i in xrange(start, len(B) - 1):
            if B[i + 1] != B[i] + 1:
                return B[i] + 1
        return B[-1] + 1


# Clone Graph
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node

    def cloneGraph(self, node):
        if node == None:
            return node
        nodeMap = {}
        return self.cloneNode(node, nodeMap)

    def cloneNode(self, node, nodeMap):
        if node == None:
            return node
        if nodeMap.has_key(node):
            return nodeMap[node]
        else:
            clone = UndirectedGraphNode(node.label)
            nodeMap[node] = clone
            for neighbor in node.neighbors:
                clone.neighbors.append(self.cloneNode(neighbor, nodeMap))
        return clone


# Valid Number
class Solution:
    # @param s, a string
    # @return a boolean

    def isNumber(self, s):
        INVALID = 0
        SPACE = 1
        SIGN = 2
        DIGIT = 3
        DOT = 4
        EXPONENT = 5
        transitionMatrix = [[-1, 0, 3, 1, 2, -1],  # 0 no input or just spaces
                            [-1, 8, -1, 1, 4, 5],  # 1 input is digits
                            # 2 no digits in front just Dot
                            [-1, -1, -1, 4, -1, -1],
                            [-1, -1, -1, 1, 2, -1],  # 3 sign
                            [-1, 8, -1, 4, -1, 5],  # 4 digits and dot in front
                            [-1, -1, 6, 7, -1, -1],  # 5 input 'e' or 'E'
                            [-1, -1, -1, 7, -1, -1],  # 6 after 'e' input sign
                            [-1, 8, -1, 7, -1, -1],  # 7 after 'e' input digits
                            [-1, 8, -1, -1, -1, -1]]  # 8 after valid input input space
        state = 0
        i = 0
        while i < len(s):
            inputType = INVALID
            if s[i] == ' ':
                inputType = SPACE
            elif s[i] == '-' or s[i] == '+':
                inputType = SIGN
            elif s[i] in '1234567890':
                inputType = DIGIT
            elif s[i] == '.':
                inputType = DOT
            elif s[i] == 'e' or s[i] == 'E':
                inputType = EXPONENT

            state = transitionMatrix[state][inputType]
            if state == -1:
                return False
            else:
                i += 1

        return state == 1 or state == 4 or state == 7 or state == 8


# Valid Palindrome
class Solution:
    # @param s, a string
    # @return a boolean

    def isPalindrome(self, s):
        if len(s) <= 1:
            return True
        s1 = self.cleanup(s.lower())
        s2 = s1[::-1]
        if s1 == s2:
            return True
        else:
            return False

    def cleanup(self, s):
        alphanumeric = "zxcvbnmlkjhgfdsaqwertyuiop1234567890"
        cleaned = ""
        for i in s:
            if i in alphanumeric:
                cleaned += i
        return cleaned


# Add Two Numbers
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @return a ListNode

    def addTwoNumbers(self, l1, l2):
        if l1 == None:
            return l2
        elif l2 == None:
            return l1

        p = l1
        q = l2
        dummy = ListNode(0)
        s = dummy
        carry = 0

        while p or q:
            if p != None:
                if q != None:
                    curr = p.val + q.val + carry
                    p = p.next
                    q = q.next
                else:
                    curr = p.val + carry
                    p = p.next
            else:
                if q != None:
                    curr = q.val + carry
                    q = q.next

            s.next = ListNode(curr % 10)
            carry = curr / 10
            s = s.next

        if carry > 0:
            s.next = ListNode(carry)
        return dummy.next


# Copy List with Random Pointer
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


# Recover Binary Search Tree
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
        list = []
        listpointer = []
        self.inorder(root, list, listpointer)
        list.sort()
        for i in xrange(len(list)):
            listpointer[i].val = list[i]
        return root

    def inorder(self, root, list, listpointer):
        if root:
            self.inorder(root.left, list, listpointer)
            list.append(root.val)
            listpointer.append(root)
            self.inorder(root.right, list, listpointer)


# Zigzag Conversion
class Solution:
    # @return a string

    def convert(self, s, nRows):
        if nRows == 1:
            return s
        tmp = ['' for i in range(nRows)]
        index = -1
        step = 1
        for i in xrange(len(s)):
            index += step
            if index == nRows:
                index -= 2
                step = -1
            elif index == -1:
                index = 1
                step = 1

            tmp[index] += s[i]
        return ''.join(tmp)


# Jump Game II
# Always try the NEXT furthest jump.
class Solution:
    # @param A, a list of integers
    # @return an integer

    def jump(self, A):
        if len(A) == 1:
            return 0

        longest = 0
        count = 0

        while True:
            count += 1
            for tmp in xrange(longest + 1):
                longest = max(longest, tmp + A[tmp])
                if longest >= len(A) - 1:
                    return count


# Distinct Subsequences
class Solution:
    # @return an integer

    def numDistinct(self, S, T):
        dp = [[0 for i in xrange(len(T) + 1)] for j in xrange(len(S) + 1)]
        # null is the substring of any string, so if the length of T len(T) is
        # 0, result is 1
        for i in xrange(len(S) + 1):
            dp[i][0] = 1

        for i in xrange(1, len(S) + 1):
            for j in xrange(1, min(i + 1, len(T) + 1)):
                if S[i - 1] == T[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]

        return dp[len(S)][len(T)]


# Combination Sum II
class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers

    def combinationSum2(self, candidates, target):
        self.res = []
        # must sort the list first in this way of solution.
        candidates.sort()
        self.recur(candidates, target, 0, [])
        return self.res

    def recur(self, candidates, target, start, ret):
        if target == 0 and ret not in self.res:
            return self.res.append(ret)
        for i in xrange(start, len(candidates)):
            if target < candidates[i]:
                return
            self.recur(
                candidates, target - candidates[i], i + 1, ret + [candidates[i]])


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a ListNode
    # @return a ListNode

    def deleteDuplicates(self, head):
        if head == None or head.next == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        q = dummy.next

        while p.next:
            while q.next and q.next.val == p.next.val:
                q = q.next
            if q == p.next:
                p = p.next
                q = p.next
            else:
                p.next = q.next

        return dummy.next


# Max Points in a Line
# Definition for a point
# class Point:
#     def __init__(self, a=0, b=0):
#         self.x = a
#         self.y = b

class Solution:
    # @param points, a list of Points
    # @return an integer

    def maxPoints(self, points):
        if len(points) < 3:
            return len(points)
        res = -1

        for i in xrange(len(points)):
            slope = {"infinite": 0}
            duplicated = 1
            for j in xrange(len(points)):
                if i == j:
                    continue
                elif points[j].x == points[i].x and points[j].y != points[i].y:
                    # slope is infinite
                    slope["infinite"] += 1
                elif points[j].x != points[i].x:
                    k = (points[j].y - points[i].y) / \
                        (points[j].x - points[i].x)
                    if k in slope:
                        slope[k] += 1
                    else:
                        slope[k] = 1
                else:
                    # duplicated points
                    duplicated += 1
            res = max(res, max(slope.values()) + duplicated)
        return res


# Permutation II
# duplicated number in list, sort first
class Solution:
    # @param num, a list of integer
    # @return a list of lists of integers

    def permuteUnique(self, num):
        if len(num) == 0:
            return []
        if len(num) == 1:
            return [num]
        num.sort()
        res = []
        prev = None
        for i in xrange(len(num)):
            if num[i] == prev:
                continue
            prev = num[i]
            for j in self.permuteUnique(num[:i] + num[i + 1:]):
                res.append(j + [num[i]])

        return res


# Gas Station
class Solution:
    # @param gas, a list of integers
    # @param cost, a list of integers
    # @return an integer

    def canCompleteCircuit(self, gas, cost):

        length = len(gas)
        sum = 0
        total = 0
        k = -1
        for i in xrange(length):
            sum += gas[i] - cost[i]
            total += gas[i] - cost[i]
            if sum < 0:
                k = i
                sum = 0

        return k + 1 if total >= 0 else -1


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
        if head == None or head.next == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        curr = head

        while curr.next:
            if curr.next.val >= curr.val:
                curr = curr.next
            else:
                pre = dummy
                while pre.next.val < curr.next.val:
                    pre = pre.next
                tmp = curr.next
                curr.next = curr.next.next
                tmp.next = pre.next
                pre.next = tmp

        return dummy.next


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

        return [end, start]


# Validate Binary Search Tree
# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param root, a tree node
    # @return a boolean

    def isValidBST(self, root):
        self.minimumNum = -10000000
        return self.solve(root)

    def solve(self, root):
        if root == None:
            return True

        if not self.solve(root.left):
            return False
        if root.val <= self.minimumNum:
            return False
        self.minimumNum = root.val
        if not self.solve(root.right):
            return False

        return True


# Next Permutation
class Solution:
    # @param num, a list of integer
    # @return a list of integer

    def nextPermutation(self, num):
        for i in xrange(len(num) - 2, -1, -1):
            if num[i] < num[i + 1]:
                flag = i
                break
        if flag == -1:
            num.reverse()
            return num
        else:
            for i in xrange(len(num) - 1, flag, -1):
                if num[i] > num[flag]:
                    num[i], num[flag] = num[flag], num[i]
                    break
        num[flag + 1:] = num[flag + 1:][::-1]
        return num


# Palindrome Partitioning
class Solution:
    # @param s, a string
    # @return a list of lists of string

    def partition(self, s):
        self.res = []
        self.dfs(s, [])
        return self.res

    def isPalindrome(self, s):
        if s == s[::-1]:
            return True
        return False

    def dfs(self, s, stringlist):
        if len(s) == 0:
            self.res.append(stringlist)
        for i in xrange(1, len(s) + 1):
            if self.isPalindrome(s[:i]):
                self.dfs(s[i:], stringlist + [s[:i]])


# n-queens
class Solution:
    # @return a list of lists of string

    def solveNQueens(self, n):
        self.res = []
        self.solve(n, 0, [-1 for i in xrange(n)])
        return self.res

    def solve(self, n, currQueen, board):
        if currQueen == n:
            oneAnsw = [['.' for i in xrange(n)] for j in xrange(n)]
            for i in xrange(n):
                oneAnsw[i][board[i]] = 'Q'
                oneAnsw[i] = ''.join(oneAnsw[i])
            self.res.append(oneAnsw)
            return
        for i in xrange(n):
            valid = True
            for k in xrange(currQueen):
                if board[k] == i:
                    valid = False
                    break
                if abs(board[k] - i) == currQueen - k:
                    valid = False
                    break
            if valid:
                board[currQueen] = i
                self.solve(n, currQueen + 1, board)


# reverse-linked-list-ii
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # @param head, a ListNode
    # @param m, an integer
    # @param n, an integer
    # @return a ListNode

    def reverseBetween(self, head, m, n):
        if head == None or head.next == None:
            return head
        dummy = ListNode(0)
        dummy.next = head

        h1 = dummy
        for i in xrange(m - 1):
            h1 = h1.next

        p = h1.next

        for i in xrange(m, n):
            tmp = h1.next
            h1.next = p.next
            p.next = p.next.next
            h1.next.next = tmp

        return dummy.next


# construct-binary-tree-from-preorder-and-inorder-traversal/

# populating-next-right-pointers-in-each-node-ii
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
        curr = root
        nextLevelHead = None
        nextLevelEnd = None
        while curr:
            if curr.left:
                if nextLevelHead:
                    nextLevelEnd.next = curr.left
                    nextLevelEnd = nextLevelEnd.next
                else:
                    nextLevelHead = curr.left
                    nextLevelEnd = nextLevelHead
            if curr.right:
                if nextLevelHead:
                    nextLevelEnd.next = curr.right
                    nextLevelEnd = nextLevelEnd.next
                else:
                    nextLevelHead = curr.right
                    nextLevelEnd = nextLevelHead
            curr = curr.next
            if not curr:
                curr = nextLevelHead
                nextLevelHead = None
                nextLevelEnd = None


# Pascal's Triangle II
class Solution:
    # @return a list of integers

    def getRow(self, rowIndex):
        count = 0

        if rowIndex == 0:
            return [1]

        list_odd = [1 for i in xrange(rowIndex + 1)]
        list_even = [1 for i in xrange(rowIndex + 1)]

        for i in xrange(2, rowIndex + 1):
            if i % 2 == 0:
                for j in xrange(1, i):
                    list_even[j] = list_odd[j - 1] + list_odd[j]
            elif i % 2 == 1:
                for j in xrange(1, i):
                    list_odd[j] = list_even[j - 1] + list_even[j]

        if rowIndex % 2 == 0:
            return list_even
        else:
            return list_odd


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
            if len(res) < level + 1:
                res.append([])
            res[level].append(root.val)
            self.preorder(root.left, level + 1, res)
            self.preorder(root.right, level + 1, res)


# 3Sum Closest
class Solution:
    # @return an integer

    def threeSumClosest(self, num, target):
        num.sort()
        mindiff = 100000
        res = 0
        for i in range(len(num)):
            left = i + 1
            right = len(num) - 1
            while left < right:
                sum = num[i] + num[left] + num[right]
                diff = abs(sum - target)
                if diff < mindiff:
                    mindiff = diff
                    res = sum
                if sum == target:
                    return sum
                elif sum < target:
                    left += 1
                else:
                    right -= 1
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
            if i % 2 == 1:
                res[i].reverse()
        return res

    def preorder(self, root, level, res):
        if root:
            if len(res) < level + 1:
                res.append([])
            res[level].append(root.val)
            self.preorder(root.left, level + 1, res)
            self.preorder(root.right, level + 1, res)


# Triangle
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
                    triangle[i][
                        j] += min(triangle[i - 1][j - 1], triangle[i - 1][j])

        return min(triangle[-1])


#Count and Say
class Solution:
    # @return a string

    def countAndSay(self, n):
        s = '1'
        for i in xrange(2, n + 1):
            s = self.count(s)
        return s

    def count(self, s):
        t = ''
        count = 0
        curr = '#'
        for i in s:
            if i != curr:
                if curr != '#':
                    t += str(count) + curr
                curr = i
                count = 1
            else:
                count += 1
        t += str(count) + curr
        return t

# Subset II


class Solution:
    # @param num, a list of integer
    # @return a list of lists of integer

    def subsetsWithDup(self, S):
        def dfs(depth, start, valuelist):
            if valuelist not in res:
                res.append(valuelist)
            if depth == len(S):
                return
            for i in xrange(start, len(S)):
                dfs(depth + 1, i + 1, valuelist + [S[i]])
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
        if l1 == None:
            return l2
        if l2 == None:
            return l1

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

        if l1 != None:
            node.next = l1
        elif l2 != None:
            node.next = l2
        return head

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
        return self.check(root)[1]

    def check(self, root):
        if root == None:
            return (0, True)
        LH, LB = self.check(root.left)
        RH, RB = self.check(root.right)
        return (max(LH, RH) + 1, LB and RB and abs(LH - RH) <= 1)

# Symmetric Tree
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
        if root == None:
            return True

        return self.check(root.left, root.right)

    def check(self, p, q):
        if p == None and q == None:
            return True
        elif p == None or q == None:
            return False

        return p.val == q.val and self.check(p.left, q.right) and self.check(p.right, q.left)


# Remove Duplicates from Sorted Array
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
                j += 1
            else:
                i += 1
                A[i] = A[j]
                j += 1
        return i + 1


class Solution:
    # @param num, a list of integer
    # @return a list of lists of integers

    def permute(self, num):
        if len(num) == 0:
            return []
        if len(num) == 1:
            return [num]
        res = []
        for i in xrange(len(num)):
            for j in self.permute(num[0:i] + num[i + 1:]):
                res.append(j + [num[i]])
        return res


# Generate Parentheses
class Solution:
    # @param an integer
    # @return a list of string

    def generateParenthesis(self, n):
        return self.helper(n, n)

    def helper(self, n, m):
        if n == 0:
            return [")" * m]
        elif n == m:
            return ["(" + i for i in self.helper(n - 1, m)]
        else:
            return ["(" + i for i in self.helper(n - 1, m)] + [")" + i for i in self.helper(n, m - 1)]


# Search in Rotated Sorted Array
class Solution:
    # @param A, a list of integers
    # @param target, an integer to be searched
    # @return an integer

    def search(self, A, target):
        l = 0
        r = len(A) - 1
        while l <= r:
            m = (l + r) // 2
            if A[m] == target:
                return m
            elif A[m] >= A[l]:
                if A[l] <= target and A[m] >= target:
                    r = m - 1
                else:
                    l = m + 1
            else:
                if A[m] <= target and A[r] >= target:
                    l = m + 1
                else:
                    r = m - 1
        return -1


# Palindrome Number
class Solution:
    # @return a boolean

    def isPalindrome(self, x):
        if x < 0:
            return False
        x_string = str(x)
        x_reverse = self.reverseStr(x_string)
        if x_string == x_reverse:
            return True
        else:
            return False

    def reverseStr(self, x_str):
        if len(x_str) <= 1:
            return x_str
        return self.reverseStr(x_str[1:]) + x_str[0]


# Sum Root to Leaf Numbers
# Timeout code:
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

    def helper(self, stack):
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

    def sum(self, root, preSum):
        if root == None:
            return 0
        preSum = root.val + 10 * preSum
        if root.left == None and root.right == None:
            return preSum
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
            for i in range(pos + 1, n + 1):
                _res.append([i])
        elif k > 1:
            while pos + k <= n:
                for x in self.getComb(res, n, pos + 1, k - 1):
                    _res.append(x)
                    _res[-1].insert(0, pos + 1)
                pos += 1
        return _res


# Subsets
class Solution:
    # @param S, a list of integer
    # @return a list of lists of integer

    def subsets(self, S):
        S.sort()
        result = [[]]
        for x in range(0, len(S) + 1):
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
                for x in self.getComb(res, S, pos + 1, k - 1):
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
        if root == None:
            return 0
        elif root.left == None and root.right == None:
            return 1
        elif root.left == None and root.right != None:
            return self.minDepth(root.right) + 1
        elif root.right == None and root.left != None:
            return self.minDepth(root.left) + 1
        elif root.left != None and root.right != None:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1


# Length of Last Word
class Solution:
    # @param s, a string
    # @return an integer

    def lengthOfLastWord(self, s):
        if len(s) == 0:
            return 0
        elif s[-1] == " ":
            s = s[:-1]
            return self.lengthOfLastWord(s)
        else:
            words = s.split(" ")
            word = words[-1]
            return len(word)


# Sudoku
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

    # small board is a 3*3 board and all num of it should be unique.
    def isValidSudokuSmall(self, smallboard):
        smallboard = smallboard.remove('.')
        if len(smallboard) == len(set(smallboard)):
            return True
        return False

    def checkLines(self, board):
        for line in range(0, 10):
            lset = board[line].remove('.')
            if len(lset) != len(set(lset)):
                return False
        for row in range(0, 10):
            rset = board[:, row].remove('.')
            if len(rset) != len(set(rset)):
                return False
        return True


# Trapping Rain Water

class Solution:
    # @param A, a list of integers
    # @return an integer

    def trap(self, A):
        if len(A) <= 2:
            return 0
        LR = self.fromLtoR(A)
        RL = self.fromRtoL(A)
        sum = 0
        for i in range(0, len(A)):
            tmp = min(LR[i], RL[i]) - A[i]
            if tmp > 0:
                sum += tmp
        return sum

    def fromLtoR(self, A):
        max = A[0]
        LR = [0] * len(A)
        for i in range(1, len(A)):
            if A[i] > max:
                max = A[i]
                LR[i] = max
            else:
                LR[i] = max
        return LR

    def fromRtoL(self, A):
        max = A[-1]
        RL = [0] * len(A)
        for i in range(len(A) - 2, -1, -1):
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
        r = len(A) - 1
        while l <= r:
            m = (r + l) // 2
            if A[m] == target:
                return True
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
        if head == None:
            return
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
        if m == 1 or n == 1:
            return 1
        for i in xrange(m):
            arr.append([])
            for j in xrange(n):
                arr[i].append(0)
        arr[0][0] = 0
        for i in xrange(1, m):
            arr[i][0] = 1
        for j in xrange(1, n):
            arr[0][j] = 1
        for i in xrange(1, m):
            for j in xrange(1, n):
                arr[i][j] = arr[i][j - 1] + arr[i - 1][j]
        return arr[-1][-1]


# Unique Paths II
class Solution:
    # @param obstacleGrid, a list of lists of integers
    # @return an integer

    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        arr = [[0 for j in xrange(n)] for i in xrange(m)]
        for j in xrange(0, n):
            if obstacleGrid[0][j] == 0:
                arr[0][j] = 1
            else:
                break
        for i in xrange(0, m):
            if obstacleGrid[i][0] == 0:
                arr[i][0] = 1
            else:
                break
        for i in xrange(1, m):
            for j in xrange(1, n):
                if obstacleGrid[i][j] == 1:
                    arr[i][j] = 0
                else:
                    arr[i][j] = arr[i][j - 1] + arr[i - 1][j]
        return arr[m - 1][n - 1]


# Valid Sudoku
class Solution:
    # @param board, a 9x9 2D array
    # @return a boolean

    def isValidSudoku(self, board):
        return self.checkRow(board) and self.checkLine(board) and self.checkBox(board)

    def checkRow(self, board):
        for row in board:
            dump = []
            for i in row:
                if i != '.':
                    if i not in dump:
                        dump.append(i)
                    else:
                        return False
        return True

    def checkLine(self, board):
        for j in xrange(9):
            dump = []
            for i in xrange(9):
                k = board[i][j]
                if k != '.':
                    if k not in dump:
                        dump.append(k)
                    else:
                        return False
        return True

    def checkBox(self, board):
        for i in xrange(3):
            for j in xrange(3):
                dump = []
                for m in xrange(3):
                    for n in xrange(3):
                        x = board[i * 3 + m][j * 3 + n]
                        if x != '.':
                            if x not in dump:
                                dump.append(x)
                            else:
                                return False
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
            num.remove(i + 1)
            i += 1
        return MCA, num

    def getMaxConsecutiveDecending(self, i, num):
        MCD = 0
        while i - 1 in num:
            MCD += 1
            num.remove(i - 1)
            i -= 1
        return MCD, num
# 解法二：


class Solution:
    # @param num, a list of integer
    # @return an integer

    def longestConsecutive(self, num):
        dict = {x: False for x in num}
        maxLen = -1
        for i in dict:
            if dict[i] == False:
                curr = i + 1
                lenAscending = 0
                while curr in dict and dict[curr] == False:
                    lenAscending += 1
                    dict[curr] = True
                    curr += 1
                curr = i - 1
                lenDecending = 0
                while curr in dict and dict[curr] == False:
                    lenDecending += 1
                    dict[curr] = True
                    curr -= 1
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
        if root == None:
            return []
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
            self.getPath(
                root.left, pathArray + [root.left.val], currSum + root.left.val)
        if root.right != None:
            self.getPath(
                root.right, pathArray + [root.right.val], currSum + root.right.val)


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
        if root == None:
            return
        self.flatten(root.left)
        self.flatten(root.right)
        if root.left == None:
            return
        else:
            p = root.left
            while p.right != None:
                p = p.right
            p.right = root.right
            root.right = root.left
            root.left = None


# Jump Game
# 解法1：（超时）
class Solution:
    # @param A, a list of integers
    # @return a boolean

    def canJump(self, A):
        if len(A) <= 1:
            return True
        B = [0] * len(A)
        for i in xrange(len(A)):
            for j in xrange(min(A[i] + 1, len(A) - i)):
                B[i + j] = max(A[i] - j, A[i + j])
        for x in B[:-1]:
            if x == 0:
                return False
        return True

# 解法2：(超时)


class Solution:
    # @param A, a list of integers
    # @return a boolean

    def canJump(self, A):
        if len(A) <= 1:
            return True
        B = [0] * len(A)
        i = 0
        while i < len(A):
            j = 0
            while j < min(A[i], len(A) - i):
                if A[i + j] > A[i] - j:
                    B[i + j] = A[i + j]
                    i = i + j - 1
                    break
                else:
                    B[i + j] = A[i] - j
                    j += 1
            i += 1
        for x in B[:-1]:
            if x == 0:
                return False
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
                if canReach >= lenA - 1:
                    return True
        return False


# Longest Common Prefix
class Solution:
    # @return a string

    def longestCommonPrefix(self, strs):
        if len(strs) == 0:
            return ""
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
        if len(A) <= 0:
            return [-1, -1]
        center = self.locate(A, target)
        if center == -1:
            return [-1, -1]
        counter_d = 0
        counter_a = 0
        while center - counter_d >= 0:
            if A[center - counter_d] == target:
                counter_d += 1
            else:
                break
        while counter_a + center < len(A):
            if A[counter_a + center] == target:
                counter_a += 1
            else:
                break
        return [center - counter_d + 1, center + counter_a - 1]

    def locate(self, A, target):
        l = 0
        s = len(A) - 1
        while l <= s:
            m = (l + s) // 2
            if target > A[m]:
                l = m + 1
            elif target < A[m]:
                s = m - 1
            else:
                return m
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
        num = []
        curr = head
        while curr != None:
            num.append(curr.val)
            curr = curr.next
        return self.treeGenerator(num, 0, len(num) - 1)

    def treeGenerator(self, num, start, end):
        if start > end:
            return None
        mid = (start + end) // 2
        root = TreeNode(num[mid])
        root.left = self.treeGenerator(num, start, mid - 1)
        root.right = self.treeGenerator(num, mid + 1, end)
        return root


# N-Queens II
class Solution:
    # @return an integer

    def totalNQueens(self, n):
        self.res = 0
        self.solve(n, 0, [-1 for i in xrange(n)])
        return self.res

    def solve(self, n, currQueenNum, board):
        if currQueenNum == n:
            self.res += 1
            return
        for i in xrange(n):
            valid = True
            for k in xrange(currQueenNum):
                if board[k] == i:
                    valid = False
                    break
                if abs(board[k] - i) == currQueenNum - k:
                    valid = False
                    break
            if valid:
                board[currQueenNum] = i
                self.solve(n, currQueenNum + 1, board)
