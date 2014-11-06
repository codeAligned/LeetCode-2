__author__ = 'leon'


def enum(**enums):
    return type('Enum', (), enums)


Alphabet = enum(BINARY='01',
                OCTAL='01234567',
                DECIMAL='0123456789',
                HEXA_DECIMAL='0123456789abcdef',
                ALPHA_LOWER='abcdefghijklmnopqrstuvwxyz',
                ALPHA_UPPER='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                ALPHA='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                ALPHA_NUMERIC='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


def convert(input, source, target):
    """BEGIN"""
