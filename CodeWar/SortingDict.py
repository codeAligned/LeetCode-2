__author__ = 'leon'


def sort_dict(d):
    'return a sorted list of tuples from the dictionary'
    res = []
    for k in d.iterkeys():
        res.append((k, d[k]))

    return sorted(res, key=lambda tuple: tuple[1], reverse=True)


dict = {1:5,3:10,2:2,6:3,8:8}
print sort_dict(dict)
