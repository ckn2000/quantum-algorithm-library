import math
import random
import numpy
from fractions import Fraction
import qConstants as qc



def equal(a, b, epsilon):
    '''Assumes that n >= 0. Assumes that a and b are both n-qbit states or n-qbit gates. Assumes that epsilon is a positive (but usually small) real number. Returns whether a == b to within a tolerance of epsilon. Useful for doing equality comparisons in the floating-point context. Warning: Does not consider global phase changes; for example, two states that are global phase changes of each other may be judged unequal. Warning: Use this function sparingly, for inspecting output and running tests. Probably you should not use it to make a crucial decision in the middle of a big computation. In past versions of CS 358, this function has not existed. I have added it this time just to streamline the tests.'''
    diff = a - b
    if len(diff.shape) == 0:
        # n == 0. Whether they're gates or states, a and b are scalars.
        return abs(diff) < epsilon
    elif len(diff.shape) == 1:
        # a and b are states.
        return sum(abs(diff)) < epsilon
    else:
        # a and b are gates.
        return sum(sum(abs(diff))) < epsilon

def uniform(n):
    '''Assumes n >= 0. Returns a uniformly random n-qbit state.'''
    if n == 0:
        return qc.one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            imags = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            psi = numpy.array([reals[i] + imags[i] * 1j for i in range(2**n)])
            psiNormSq = numpy.dot(numpy.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm

def bitValue(state):
    '''Given a one-qbit state assumed to be exactly classical --- usually because a classical state was just explicitly assigned to it --- returns the corresponding bit value 0 or 1.'''
    if (state == qc.ket0).all():
        return 0
    else:
        return 1

def powerMod(k, l, m):
    '''Given non-negative integer k, non-negative integer l, and positive integer m. Computes k^l mod m. Returns an integer in {0, ..., m - 1}.'''
    kToTheL = 1
    curr = k
    while l >= 1:
        if l % 2 == 1:
            kToTheL = (kToTheL * curr) % m
        l = l // 2
        curr = (curr * curr) % m
    return kToTheL

def toComplex(r,t):
    x = r*numpy.cos(t)
    y = r*numpy.sin(t)
    chi = x+y*1j
    return chi

def continuedFraction(n, m, x0):
    ''' x0 is a float in [0,1). Tries probing depths j = 0, 1, 2, ... until the resulting rational approximation x0 ~ c / d satisfies either d >= m or | x0 - c / d| <= 1 / 2^(n + ). Returns a pair (c,d) with gcd(c, d) = 1.'''
    def fraction(x, j):
        if x == 0:
            return 0
        else:
            if j == 1:
                return Fraction(1,math.floor(1/x))
            else:
                return Fraction(1,math.floor(1/x)+fraction(1/x-math.floor(1/x), j-1))
    
    j = 1
    if Fraction(x0).numerator == 1 or x0 == 0:
        return Fraction(x0)
    else:
        while ((fraction(x0,j).denominator < m) or (abs(x0-fraction(x0,j)) > 1/2**(n+1))):
            j += 1
        return fraction(x0,j)

def directSum(A,B) :
    C = numpy.zeros((A.shape[0]+B.shape[0],A.shape[1]+B.shape[1]), dtype=numpy.array(0+0j).dtype)
    for r in range(C.shape[0]) :
        for c in range(C.shape[1]) :
            try:
                C[r,c] = A[r,c]
            except IndexError:
                C[r,c] = 0
    
    for r in range(-1,-C.shape[0]+1,-1) :
        for c in range(-1,-C.shape[1]+1,-1) :
            try:
                C[r,c] = B[r,c]
            except IndexError:
                C[r,c] = 0
    return C