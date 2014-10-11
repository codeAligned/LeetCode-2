__author__ = 'leon'

import os


def mkdirp(*directories):
    """Recursively create all directories as necessary"""
    direct = [d for d in directories]
    lendirect = len(direct)
    for i in xrange(1, lendirect + 1):
        tmpdir = '/'.join(direct[:i])
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)


mkdirp('tmp', 'a', 'b')