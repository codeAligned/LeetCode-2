import sys


def count_calls(func, *args, **kwargs):
    """Count calls in function func"""

    calls = [-1]

    def tracer(frame, event, arg):
        if event == 'call':
            calls[0] += 1
        return tracer

    sys.settrace(tracer)

    rv = func(*args, **kwargs)

    return calls[0], rv


def add(a, b):
    return a + b


def add_ten(a):
    return add(a, 10)


def misc_fun():
    return add(add_ten(3), add_ten(9))


print count_calls(misc_fun)