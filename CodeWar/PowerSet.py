__author__ = 'leon'


def power(s):
    """Computes all of the sublists of s"""

    def dfs(valuelist, depth, start):
        res.append(valuelist)
        if depth == len(s): return
        for i in xrange(start, len(s)):
            dfs(valuelist + [s[i]], depth + 1, i + 1)

    s.sort()
    res = []
    dfs([], 0, 0)
    return res