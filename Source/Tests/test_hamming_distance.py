from Source.germinal_center import *

def test1():
    assert hamming_distance(1000, 1000) == 0

def test2():
    assert hamming_distance(1000, 1001) == 1

def test3():
    assert hamming_distance(1234, 5678) == 16

"""
This case will never occur as the value cannot be '0000'.
def test4():
    assert hamming_distance(9999, 0000) == 36
"""

def test5():
    assert hamming_distance(5491, 2941) == 13