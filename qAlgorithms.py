import numpy
import random
import math
from scipy.linalg import lu

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qMeasurement as qm
import qBitStrings as qb




def bennett():
    '''Runs one iteration of the core algorithm of Bennett (1992). Returns a tuple of three items --- |alpha>, |beta>, |gamma> --- each of which is either |0> or |1>.'''
    # Aiko chooses ketAlpha
    basisChoice = [qc.ket0, qc.ket1]
    basisWeight = [0.5,0.5]
    ketAlpha = random.choices(basisChoice, basisWeight)[0]

    # Aiko sends ketPsi
    if (ketAlpha == qc.ket0).all() :
        ketPsi = qc.ket0
    else :
        ketPsi = qc.ketPlus

    # Babatope chooses ketBeta
    ketBeta = random.choices(basisChoice, basisWeight)
    
    # Babatope sends ketGamma
    if (ketBeta == qc.ket0).all() :
        ketGamma = qm.first((qg.tensor(ketPsi, qc.ket0)))[0]
    
    elif (ketBeta == qc.ket1).all() :
        ketGamma = qm.first((qg.tensor(qg.application(qc.h, ketPsi), qc.ket0)))[0]

    return (ketAlpha, ketBeta, ketGamma)
    
def deutsch(f):
    '''Implements the algorithm of Deutsch (1985). That is, given a two-qbit gate F representing a function f : {0, 1} -> {0, 1}, returns |1> if f is constant, and |0> if f is not constant.'''
    # input is two |1>'s
    input = qg.tensor(qc.ket1, qc.ket1)
    # goes through a layer of 2 Hadamard gates
    firstH = qg.application(qg.tensor(qc.h, qc.h), input)
    # goes through the 2-qbit gate F
    gateF = qg.application(f, firstH)
    # goes through the second layer of 2 Hadamard gates
    secondH = qg.application(qg.tensor(qc.h, qc.h), gateF)

    # returns the measurement of the resulting state
    return qm.first(secondH)[0]

def bernsteinVazirani(n,f):
    ''' Given n>= 1 and an (n+1)-qbit gate F representing a function f : {0,1}^n -> {0,1} defined by mod-2 dot product with an unknown delta in {0,1}^n,
    returns the list or tuple of n classical one-qbit states (each |0> or |1> corresponding to delta.'''
    # input is n-qbit |0> and 1-qbit |1>
    inputK = qg.tensor(qg.power(qc.ket0,n), qc.ket1)
    # goes through the first layer of n+1 Hadamard gates
    firstH = qg.application(qg.power(qc.h,n+1), inputK)
    # goes through the (n+1)-qbit F gate
    gateF = qg.application(f, firstH)
    # goes through the second layer of n+1 Hadamard gates
    secondH = qg.application(qg.power(qc.h,n+1), gateF)

    measurementTuple = ()

    # measure the input register
    for i in range(n):
        measurement = qm.first(secondH)
        measurementTuple += (measurement[0],)
        secondH = measurement[1]

    # return the measurement result
    return measurementTuple

def simon(n,f):
    '''The inputs are an integer n >= 2 and an (n + (n - 1))-qbit gate F representing a function f: {0, 1}^n -> {0, 1}^(n - 1) hiding an n-bit string delta as in the Simon (1994) problem. Returns a list or tuple of n
    classical one-qbit states (each |0> or |1>) corresponding to a uniformly random bit string gamma that is perpendicular to delta.'''
    # input register is n-qbit |0>
    inputK = qg.power(qc.ket0,n)
    # input register goes through a layer of Hadamard gates; output register is (n-1)-qbit |0> which stays the same
    firstH = qg.tensor(qg.application(qg.power(qc.h,n),inputK), qg.power(qc.ket0, n-1))
    # input and output registers go through the (n + (n - 1))-qbit gate F
    gateF = qg.application(f, firstH)

    # measures the last (n-1) qbits (output register) to unentangle them from the first n qbits (input register)
    for i in range(n-1):
        measurement = qm.last(gateF)
        gateF = measurement[0]
    
    # input register goes through a second layer of Hadamard gates
    secondH = qg.application(qg.power(qc.h,n), gateF)
    measurementTuple = ()

    # measures the input register
    for i in range(n):
        measurementInput = qm.first(secondH)
        measurementTuple += (measurementInput[0],)
        secondH = measurementInput[1]

    # returns measurement
    return measurementTuple

def shor(n,f):
    ''' Assume n>= 1. Given an (n+n)-qbit gate F representing a function f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list or tuple of n classical one q-bit states (|0> or |1>) corresponding to the output of Shor's quantum circuit'''
    # input register is n-qbit |0>
    inputRegister = qg.power(qc.ket0, n)
    # input register goes through a layer of Hadamard gates
    inputResFirstH = qg.application(qg.power(qc.h, n), inputRegister)
    # output register is n-qbit |0>, does not go through a layer of Hadamard gates
    # this is the state of the system after going through the first layer of H gates
    postH = qg.tensor(inputResFirstH, qg.power(qc.ket0,n))
    # goes through the F gate
    postF = qg.application(f, postH)

    # measure the output register to unentangle it from the input register
    for i in range(n):
        outMeasure = qm.last(postF)
        postF = outMeasure[0]
    
    # input register goes through the QFT gate
    fourierGate = qg.application(qg.fourier(n), postF)
    measurementTuple = ()

    # measure the input register
    for j in range(n):
        inMeasure = qm.first(fourierGate)
        measurementTuple += (inMeasure[0],)
        fourierGate = inMeasure[1]
    
    # return the result of the measurement
    return measurementTuple

def grover(n, k, f):
    ''' Assume n>=1, k>=1. Assumes that k is small compared to 2^n. Implements the Grover core subroutine. The F parameter is an (n+1)-qbit gate representing a function f : {0, 1}^n -> {0, 1}
    such that SUM_alpha f(alpha) = k. Returns a list or tuple of n classical one-qbit states (either |0> or |1>, such that the corresponding n-bit string delta usually satisfies f(delta) = 1.'''
    
    # ketRho
    ketRho = qg.power(qc.ketPlus,n)
    braRho = numpy.conj(numpy.transpose(ketRho))

    # R matrix
    Rmatrix = 2*(numpy.outer(ketRho, braRho))-qg.power(qc.i,n)
    RtensorI = qg.tensor(Rmatrix, qc.i)

    # rotation markers
    t = numpy.arcsin((k**(1/2))*(2**(-n/2)))
    l = round(numpy.pi/(4*t)-1/2)
    
    # input register
    ketInput = qg.tensor(qg.power(qc.ket0,n), qc.ket1)
    # goes through a layer of Hadamard gates
    firstH = qg.application(qg.power(qc.h,n+1), ketInput)

    # apply the rotation gates l times
    for i in range(l):
        gateF = qg.application(f, firstH)
        firstH = qg.application(RtensorI, gateF)
    
    measurementTuple = ()

    # measures the input register
    for i in range(n):
        measurementInput = qm.first(firstH)
        measurementTuple += (measurementInput[0],)
        firstH = measurementInput[1]

    # returns result of the measurement
    return measurementTuple


### DEFINING SOME TESTS ###

def bennettTest(m):
    # Runs Bennett's core algorithm m times.
    trueSucc = 0
    trueFail = 0
    falseSucc = 0
    falseFail = 0
    for i in range(m):
        result = bennett()
        if qu.equal(result[2], qc.ket1, 0.000001):
            if qu.equal(result[0], result[1], 0.000001):
                falseSucc += 1
            else:
                trueSucc += 1
        else:
            if qu.equal(result[0], result[1], 0.000001):
                trueFail += 1
            else:
                falseFail += 1
    print("check bennettTest for false success frequency exactly 0")
    print("    false success frequency = ", str(falseSucc / m))
    print("check bennettTest for true success frequency about 0.25")
    print("    true success frequency = ", str(trueSucc / m))
    print("check bennettTest for false failure frequency about 0.25")
    print("    false failure frequency = ", str(falseFail / m))
    print("check bennettTest for true failure frequency about 0.5")
    print("    true failure frequency = ", str(trueFail / m))

def deutschTest():
    def fNot(x):
        return (1 - x[0],)
    resultNot = deutsch(qg.function(1, 1, fNot))
    if qu.equal(resultNot, qc.ket0, 0.000001):
        print("passed deutschTest first part")
    else:
        print("failed deutschTest first part")
        print("    result = " + str(resultNot))
    def fId(x):
        return x
    resultId = deutsch(qg.function(1, 1, fId))
    if qu.equal(resultId, qc.ket0, 0.000001):
        print("passed deutschTest second part")
    else:
        print("failed deutschTest second part")
        print("    result = " + str(resultId))
    def fZero(x):
        return (0,)
    resultZero = deutsch(qg.function(1, 1, fZero))
    if qu.equal(resultZero, qc.ket1, 0.000001):
        print("passed deutschTest third part")
    else:
        print("failed deutschTest third part")
        print("    result = " + str(resultZero))
    def fOne(x):
        return (1,)
    resultOne = deutsch(qg.function(1, 1, fOne))
    if qu.equal(resultOne, qc.ket1, 0.000001):
        print("passed deutschTest fourth part")
    else:
        print("failed deutschTest fourth part")
        print("    result = " + str(resultOne))

def bernsteinVaziraniTest(n):
    delta = qb.string(n, random.randrange(0, 2**n))
    def f(s):
        return (qb.dot(s, delta),)
    gate = qg.function(n, 1, f)
    qbits = bernsteinVazirani(n, gate)
    bits = tuple(map(qu.bitValue, qbits))
    diff = qb.addition(delta, bits)
    if diff == n * (0,):
        print("passed bernsteinVaziraniTest")
    else:
        print("failed bernsteinVaziraniTest")
        print("    delta = " + str(delta))

def simonTest(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Build a certain matrix M.
    k = 0
    while delta[k] == 0:
        k += 1
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it is a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    
    trivialSolution = tuple([0 for i in range(n)])
    gammaMatrix = []
    while len(gammaMatrix) != n-1:
        gamma = simon(n,gate)
        gammaBitString = ()
        for i in range(n):
            gammaBitString += qb.ketToBitString(gamma[i],1)
        gammaMatrix.append(gammaBitString)
        gammaMatrix = qb.reduction(gammaMatrix)
        for i in range(len(gammaMatrix)):
            if gammaMatrix[i] == trivialSolution:
                del gammaMatrix[i]

    # find column with free variable
    U = lu(gammaMatrix)[2]
    basis = {numpy.flatnonzero(U[i, :])[0] for i in range(U.shape[0])}
    free_variable = list(set(range(U.shape[1])) - basis)[0]

    rowWithFreeVariable = []
    solution = [0 for i in range(n-1)]

    # find row of matrix that contains free variables
    for i in range(len(gammaMatrix)):
        if gammaMatrix[i][free_variable] == 1:
            rowWithFreeVariable.append(i)
    # changed solution to resize matrix
    for index in rowWithFreeVariable:
        solution[index] = 1

    gammaMatrixSquare = gammaMatrix.copy()
    # resize matrix to square matrix 
    gammaMatrixSquare = numpy.delete(gammaMatrixSquare, free_variable, axis=1)
    gammaMatrixSquare = numpy.array(gammaMatrixSquare)
    solution = numpy.array(solution)

    # solve for matrix
    prediction = list((numpy.linalg.solve(gammaMatrixSquare, solution)).astype(int))
    prediction.insert(free_variable, 1)
    prediction = tuple(prediction)

    if delta == prediction:
        print("passed simonTest")
    else:
        print("failed simonTest")
        print("     delta = " + str(delta))
        print("     prediction = " + str(prediction))

def shorTest(n,m):
    d = m
    dPrime = m

    # Choose a random k
    k = 0
    while (math.gcd(k,m) != 1):
        k = random.randint(0,m-1)
    def f(l):
        return qb.string(n, qu.powerMod(k,qb.integer(l),m))
    gate = qg.function(n,n,f)

    while d>=m:
        output = shor(n,gate)
        resultingBitString = ()

        for ket in output:
            respectiveBitString = qb.ketToBitString(ket,1)
            resultingBitString += respectiveBitString
        
        b = qb.integer(resultingBitString) % 2**n

        if qu.continuedFraction(n,m,b/(2**n)).numerator == 1:
            d = qu.continuedFraction(n,m,b/(2**n)).denominator
            break
        d = qu.continuedFraction(n,m,b/2**n).denominator
    
    if qu.powerMod(k,d,m) == 1:
        p = d
    else:
        while dPrime >= m:
            outputPrime = shor(n,gate)
            resultingBitStringPrime = ()

            for ket in outputPrime:
                respectiveBitStringPrime = qb.ketToBitString(ket,1)
                resultingBitStringPrime += respectiveBitStringPrime
            
            b = qb.integer(resultingBitStringPrime) % 2**n

            if qu.continuedFraction(n,m,b/(2**n)).numerator == 1:
                dPrime = qu.continuedFraction(n,m,b/(2**n)).denominator
                break
            dPrime = qu.continuedFraction(n,m,b/2**n).denominator
        
        if qu.powerMod(k,dPrime,m) == 1:
            p = dPrime
        else:
            lcm = (d*dPrime)/math.gcd(d,dPrime)
            if qu.powerMod(k,lcm,m) == 1:
                p = lcm
            else:
                shorTest(n,m)
                return
    
    # Individual test
    modEquiv = 0
    power = 1
    period = 0
    while modEquiv != 1:
        modEquiv = qu.powerMod(k,power,m)
        power += 1
        period += 1

    if period == p:
        print("passed shorTest")
        #print("     period = " + str(period))
        #print("     prediction = " + str(p))
    else:
        print("failed shorTest")
        print("     period = " + str(period))
        print("     prediction = " + str(p))
    
def groverTest(n, k):
    # Pick k distinct deltas uniformly randomly.
    deltas = []
    while len(deltas) < k:
        delta = qb.string(n, random.randrange(0, 2**n))
        if not delta in deltas:
            deltas.append(delta)
    # Prepare the F gate.
    def f(alpha):
        for delta in deltas:
            if alpha == delta:
                return (1,)
            return (0,)
    fGate = qg.function(n, 1, f)
    # Run Grovers algorithm up to 10 times.
    qbits = grover(n, k, fGate)
    bits = tuple(map(qu.bitValue, qbits))
    j = 1
    while (not bits in deltas) and (j < 10):
        qbits = grover(n, k, fGate)
        bits = tuple(map(qu.bitValue, qbits))
        j += 1
    if bits in deltas:
        print("passed groverTest in " + str(j) + " tries")

    else:
        print("failed groverTest")
        print(" exceeded 10 tries")
        print(" prediction = " + str(bits))
        print(" deltas = " + str(deltas))

### RUNNING THE TESTS ###

def main():
    bennettTest(100000)
    deutschTest()
    bernsteinVaziraniTest(3)
    simonTest(3)
    shorTest(6,6)
    groverTest(6,6)

if __name__ == "__main__":
    main()


