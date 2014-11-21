class NumStr(object):
    def __init__(self, num=0, str=''):
        self.__num = num
        self.__str = str

    def __str__(self, ):
        return '[%d::%r]' % (self.__num, self.__str)


    def __add__(self, other):
        if isinstance(other, NumStr):
            return self.__class__(self.__num + other.__num, self.__str + other.__str)
        else:
            raise TypeError, 'Illegal Argument!'


a = NumStr(1, 'AAA')
b = NumStr(2, 'BBB')

print a + b