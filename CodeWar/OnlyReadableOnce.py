__author__ = 'leon'
import unittest


class SecureList():
    def __init__(self, vallist):
        self.vallist = vallist[:]

    def __getitem__(self, item):
        tmp = self.vallist[item]
        self.vallist.pop(item)
        return tmp

    def __repr__(self):
        tmp, self.vallist = self.vallist[:], []
        return repr(tmp)

    def __str__(self):
        tmp, self.vallist = self.vallist[:], []
        return str(tmp)

    def __len__(self):
        return len(self.vallist)