#
# 28. return the number of different ways to climb stairs, each level can move
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
# 27. Interview question 1
# change the nth digit of binary value of an integer to 1


def solution(a, offset):
    tmp = a >> (offset - 1)
    if tmp % 2 == 1:
        return a
    elif tmp % 2 == 0:
        xorVal = 1 << (offset - 1)
        return a ^ xorVal
