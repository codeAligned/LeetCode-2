__author__ = 'leon'

import math


class Vector:
    def __init__(self, v=[]):
        self.vect = [i for i in v]
        self.length = len(self.vect)

    def add(self, u):
        try:
            res = []
            for i in xrange(self.length):
                res.append(self.vect[i] + u.vect[i])
            return Vector(res)
        except:
            print "error"


    def subtract(self, u):
        try:
            res = []
            for i in xrange(self.length):
                res.append(self.vect[i] - u.vect[i])
            return Vector(res)
        except:
            print "error"

    def dot(self, u):
        try:
            res = 0
            for i in xrange(self.length):
                res += self.vect[i] * u.vect[i]
            return res
        except:
            print "error"

    def norm(self):
        sum = 0
        for i in self.vect:
            sum += pow(i, 2)
        return math.sqrt(sum)

    def equals(self, u):
        try:
            for i in xrange(self.length):
                if self.vect[i] != u.vect[i]:
                    return False
            return True
        except:
            return False

    def __str__(self):
        return ''.join(str(tuple(self.vect)).split())


a = Vector([1, 2])
b = Vector([3, 4])
c = Vector([5, 6])
d = Vector([1, 2])

print a.add(b)
print a.subtract(b)
print a.dot(b)
print a.norm()

print a.equals(c)
print a.equals(d)