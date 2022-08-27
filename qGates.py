import numpy
import random
import math

import qConstants as qc
import qUtilities as qu
import qBitStrings as qb

def application(u, ketPsi):
    '''Assumes n >=1. Applies the n-qbit gate U to the n-qbit state |psi>, returning the n-qbit state U |psi>.'''
    return numpy.dot(u, ketPsi)

def tensor(a, b):
    '''Assumes that n, m >= 1. Assumes that a is an n-qbit state and b is an m-qbit state, or that a is an n-qbit gate and b is an m=qbit gate. Returns the tensor product of a and b, which is an (n+m)-qbit gate or state.'''
    # a and b are n-qbit and m-qbit states respectively
    # If a and b are both states
    if (len(a.shape) == 1 and len(b.shape) == 1) :
        c = numpy.zeros((a.shape[0]*b.shape[0],), dtype=numpy.array(0 + 0j).dtype)
        indexC = 0
        while indexC != c.shape[0]:
            for i in range(a.shape[0]) :
                for j in range(b.shape[0]) :
                    c[indexC] = a[i]*b[j]
                    indexC += 1
        return c

    # If a and b are both gates
    else:
        c = numpy.zeros((b.shape[0],a.shape[1]*b.shape[1]), dtype=numpy.array(0 +0j).dtype)
        for i in range(a.shape[0]):
            d = numpy.zeros((b.shape[0],b.shape[1]), dtype=numpy.array(0 +0j).dtype)
            for j in range(a.shape[1]):
                if j == 0:
                    d = d+a[i,j]*b
                else:
                    d = numpy.concatenate((d,a[i,j]*b), axis=1)
            if i == 0:
                c = c+d
            else:
                c = numpy.concatenate((c,d), axis=0)

        return c


def function(n, m, f):
    ''' Assumes n, m >= 1. Given a Python function f : {0, 1}^n -> {0, 1}^m.
    That is, f takes as an input an n-bit string and produces as output an m-bit string, as defined in qBitStrings.py. Returns the corresponding (n+m)-qbit gate F'''
    columnTuple = ()
    alphaBeta = tuple([0 for i in range(n+m)])
    zeroBitString = tuple([0 for i in range(n+m)])
    alpha = tuple([0 for i in range(n)])
    beta = tuple([0 for i in range(m)])
    ketAlpha = qb.bitStringToKet(alpha)
    column = tensor(ketAlpha, qb.bitStringToKet(qb.addition(beta, f(alpha))))
    columnTuple += (column,)

    alphaBeta = qb.next(alphaBeta)
    while (alphaBeta != zeroBitString):
        alpha = alphaBeta[:n]
        ketAlpha = qb.bitStringToKet(alpha)
        beta = alphaBeta[n:]

        column = tensor(ketAlpha, qb.bitStringToKet(qb.addition(beta, f(alpha))))
        columnTuple += (column,)
        alphaBeta = qb.next(alphaBeta)
    
    return numpy.column_stack(columnTuple)
    
def power(stateOrGate, m) :
    ''' Assumes n>= 1. Given an n-qbit gate or state and m >= 1, returns the mth tensor power, which is an (n * m)-qbit gate or state.
    For the sake of time and memory, m should be small.'''
    if m == 0:
        return stateOrGate
    elif m == 1 :
        return stateOrGate
    else :
        return tensor(power(stateOrGate, m-1), stateOrGate)

def fourier(n):
    ''' Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.'''
    T = numpy.zeros((2**n,2**n), dtype=numpy.array(0 +0j).dtype)
    for i in range(2**n):
        for j in range(2**n):
            T[i,j] = qu.toComplex(1/(2**(n/2)), (2*numpy.pi*i*j)/(2**n))

    return T

def fourierRecursive(n):
    ''' Assumes n>=1. Returns the n-qbit Fourier transform gate T.
    Computes T recursively rather than from the definition. '''
    if n == 1:
        return qc.h
    else:
        return application(application(fourierQ(n), fourierR(n)), fourierS(n))

def fourierS(n):
    ''' Assumes n>=1. Returns the n-qbit gate S such that S takes the last bit in an n-bit string and moves it to the first bit. '''
    firstBitString = qb.string(n,0)
    columnTuples = ()
    columnTuples += (qb.bitStringToKet(firstBitString),)
    bitString = qb.next(firstBitString)
    while bitString != firstBitString:
        resultingBitString = (bitString[len(bitString)-1],) + bitString[:len(bitString)-1]
        column = (qb.bitStringToKet(resultingBitString),)
        columnTuples += column
        bitString = qb.next(bitString)
    
    return numpy.column_stack(columnTuples)

def fourierR(n):
    ''' Assumes n>=1. Returns the n-qbit gate R. '''
    if n == 1:
        return qc.i
    else:
        gateR = tensor(qc.i, fourierRecursive(n-1))
        return gateR

def fourierQ(n):
    ''' Assumes n>=1. Returns the n-qbit gate Q. '''
    qFirstHalf = numpy.concatenate((power(qc.i,n-1),fourierD(n-1)), axis=1)
    qSecondHalf = numpy.concatenate((power(qc.i,n-1),-1*fourierD(n-1)), axis=1)
    gateQ = numpy.concatenate((qFirstHalf,qSecondHalf), axis=0)
    gateQ = (1/numpy.sqrt(2))*gateQ

    return gateQ

def fourierD(n):
    ''' Assumes n>=1. Returns the n-qbit gate D. '''
    gateD = numpy.zeros((2**n, 2**n), dtype=numpy.array(0 + 0j).dtype)
    omegaKplus1 = numpy.exp((2*numpy.pi*1j)/(2**(n+1)))

    for i in range(2**n):
        gateD[i,i] = omegaKplus1**i

    return gateD

def distant(gate):
    ''' Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is n-qbit), performs swaps to
    insert one extra wire between the first qbit and the other n qbits. Returns an (n+2)-qbit gate '''
    n = math.log2(gate.shape[0])-1
    firstSwap = tensor(qc.swap, power(qc.i,n))
    Vgate = tensor(qc.i, gate)
    postVgate = application(firstSwap, Vgate)
    circuit = application(postVgate, firstSwap)

    return circuit

def ccNot():
    ''' Returns the three-qbit ccNOT (i.e. Toffoli) gate. The gate is implemented using five specific
    two-qbit gates and some SWAPs. '''
    v = numpy.array([
        [1 + 0j, 0 + 1j],
        [0 - 1j, -1 + 0j]])
    v = (1/(numpy.sqrt(2)))*v
    u = numpy.array([
        [1 + 0j, 0 + 0j],
        [0 + 0j, 0 - 1j]])
    cV = qu.directSum(qc.i, v)
    cZ = qu.directSum(qc.i, qc.z)
    cU = qu.directSum(qc.i, u)
    uz = application(tensor(cU, qc.i), tensor(qc.i, cZ))
    uzv = application(uz, distant(cV))
    uzvz = application(uzv, tensor(qc.i, cZ))
    toffoliGate = application(uzvz, distant(cV))

    return toffoliGate

def groverR3():
    ''' Assumes that n = 3. Returns -R, where R is Grover's n-qbit gate for reflection across |rho>.
    Builds the gate from one- and two-qbit gates, rather than manually constructing the matrix.'''
    hxGate = application(power(qc.h,3), power(qc.x,3))
    xhGateccNOT = application(hxGate, tensor(power(qc.i,2), qc.h))
    xGateccNot = application(xhGateccNOT, ccNot())
    xccNOT = application(xGateccNot, tensor(power(qc.i,2), qc.h))
    xhGateccNOTx = application(xccNOT, power(qc.x,3))
    groverR = application(xhGateccNOTx, power(qc.h,3))

    return groverR

### DEFINING SOME TESTS ###

def applicationTest():
    # These simple tests detect type errors but not much else.
    answer = application(qc.h, qc.ketMinus)
    if qu.equal(answer, qc.ket1, 0.000001):
        print("passed applicationTest first part")
    else:
        print("FAILED applicationTest first part")
        print("    H |-> = " + str(answer))
    ketPsi = qu.uniform(2)
    answer = application(qc.swap, application(qc.swap, ketPsi))
    if qu.equal(answer, ketPsi, 0.000001):
        print("passed applicationTest second part")
    else:
        print("FAILED applicationTest second part")
        print("    |psi> = " + str(ketPsi))
        print("    answer = " + str(answer))

def tensorTest():
    # Pick two gates and two states.
    u = qc.x
    v = qc.h
    ketChi = qu.uniform(1)
    ketOmega = qu.uniform(1)
    # Compute (U tensor V) (|chi> tensor |omega>) in two ways.
    a = tensor(application(u, ketChi), application(v, ketOmega))
    b = application(tensor(u, v), tensor(ketChi, ketOmega))
    # Compare.
    if qu.equal(a, b, 0.000001):
        print("passed tensorTest first part")
    else:
        print("FAILED tensorTest first part")
        print("    a = " + str(a))
        print("    b = " + str(b))
    
def functionTest(n,m):
    # 2^n times, randomly pick an m-bit string.
    values = [qb.string(m, random.randrange(0, 2**m)) for k in range(2**n)]

    # Define f by using those values as a look-up table.
    def f(alpha):
        a = qb.integer(alpha)
        return values[a]

    # Build the corresponding gate F.
    ff = function(n,m,f)

    # Helper functions --- necessary because of poor planning.
    def g(gamma):
        if gamma == 0:
            return qc.ket0
        else:
            return qc.ket1
    
    def ketFromBitString(alpha):
        ket = g(alpha[0])
        for gamma in alpha[1:]:
            ket = tensor(ket, g(gamma))
        return ket

    # Check 2^n -1 value somewhat randomly.
    alphaStart = qb.string(n, random.randrange(0, 2**n))
    alpha = qb.next(alphaStart)
    while alpha != alphaStart:
        # Pick a single random beta to test against this alpha.
        beta = qb.string(m, random.randrange(0, 2**m))
        # Compute |alpha> tensor |beta + f(alpha)>.
        ketCorrect = ketFromBitString(alpha + qb.addition(beta, f(alpha)))
        # Compute F * (|alpha> tensor |beta>).
        ketAlpha = ketFromBitString(alpha)
        ketBeta = ketFromBitString(beta)
        ketAlleged = application(ff, tensor(ketAlpha, ketBeta))
        # Compare.
        if not qu.equal(ketCorrect, ketAlleged, 0.000001):
            print("failed functionTest")
            print(" alpha = " + str(alpha))
            print(" beta = " + str(beta))
            print(" ketCorrect = " + str(ketCorrect))
            print(" ketAlleged = " + str(ketAlleged))
            print(" and here is F...")
            print(ff)
            return
        else:
            alpha = qb.next(alpha)
    print("passed functionTest")

def fourierTest(n):
    if n == 1:
        # Explicitly check the answer.
        t = fourier(1)
        if qu.equal(t, qc.h, 0.000001):
            print("passed fourierTest")
        else:
            print("failed fourierTest")
            print("    got T = ...")
            print(t)
    
    else:
        t = fourier(n)
        # Check the first row and column.
        const = pow(2, -n / 2) + 0j
        for j in range(2**n):
            if not qu.equal(t[0,j], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        for i in range(2**n):
            if not qu.equal(t[i, 0], const, 0.000001):
                print("failed fourierTest first part")
                print("    t = ")
                print(t)
                return
        print("passed fourierTest first part")
        # Check that T is unitary.
        tStar = numpy.conj(numpy.transpose(t))
        tStarT = numpy.matmul(tStar, t)
        id = numpy.identity(2**n, dtype=qc.one.dtype)
        if qu.equal(tStarT, id, 0.000001):
            print("passed fourierTest second part")
        else:
            print("failed fourierTest second part")
            print("    T^* T = ...")
            print(tStarT)

def fourierRecursiveTest(n):
    fourierGate = fourier(n)
    fourierGateRecursive = fourierRecursive(n)
    if qu.equal(fourierGate, fourierGateRecursive, 0.000001):
        print("passed fourierRecursiveTest")
    else:
        print("failed fourierRecursiveTest")
        print("    T = " + str(fourierGate))
        print("    T recursive = " + str(fourierGateRecursive))

def distantTest():
    u = tensor(qc.i,qc.x)
    insertedGate = distant(u)
    realInsertedGate = tensor(qc.i, u)
    if qu.equal(insertedGate, realInsertedGate, 0.000001):
        print("passed distantTest")
    else:
        print("failed distantTest")
        print("    insertedGate = " + str(insertedGate))
        print("    realInsertedGate = " + str(realInsertedGate))

def ccNotTest():
    # the toffoli gate through computation
    toffoliGate = power(qc.i, 3)
    toffoliGate[6,6] = 0
    toffoliGate[6,7] = 1
    toffoliGate[7,6] = 1
    toffoliGate[7,7] = 0

    if qu.equal(ccNot(), toffoliGate, 0.000001):
        print("passed ccNotTest")
    else:
        print("failed ccNotTest")
        print("    ccNot computed = " + str(ccNot()))
        print("    toffoliGate = " + str(toffoliGate))

def groverR3Test():
    # ketRho
    ketRho = power(qc.ketPlus,3)
    braRho = numpy.conj(numpy.transpose(ketRho))

    # R matrix
    Rmatrix = power(qc.i,3) - 2*(numpy.outer(ketRho, braRho))
    R3grover = groverR3()

    if qu.equal(R3grover, Rmatrix, 0.000001):
        print("passed groverR3Test")
    else:
        print("failed groverR3Test")
        print("    groverR3 computed = " + str(R3grover))
        print("    -R = " + str(Rmatrix))


### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()
    functionTest(1,2)
    fourierTest(7)
    fourierRecursiveTest(4)
    distantTest()
    ccNotTest()
    groverR3Test()

if __name__ == "__main__":
    main()
