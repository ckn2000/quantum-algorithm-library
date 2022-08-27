import random
import numpy

import qConstants as qc
import qUtilities as qu
import qGates as qg



def first(state):
    '''Assumes n>=1. Given an n-qbit state, measures the first qbit. Returns a pair (tuple of two items) consisting of a classical one-qbit state (either |0> or |1>) and an (n - 1)-qbit state.'''
    sigma_0 = 0
    sigma_1 = 0
    for i in range(int(state.shape[0]/2)) :
        sigma_0 += abs(state[i])**2
    
    for j in range(int(state.shape[0]/2), int(state.shape[0])) :
        sigma_1 += abs(state[j])**2

    sigma_0 = numpy.sqrt(sigma_0)
    sigma_1 = numpy.sqrt(sigma_1)

    # Assuming the general cases where sigma values are not 0
    if (sigma_0 != 0 and sigma_1 != 0) :
        ketChi = state[:int(state.shape[0]/2)]/sigma_0
        ketPhi = state[int(state.shape[0]/2):]/sigma_1
        states = [(qc.ket0, ketChi), (qc.ket1, ketPhi)]
        weights = [abs(sigma_0)**2, abs(sigma_1)**2]

        results = random.choices(states, weights)
        return results[0]
    
    # Otherwise, if sigma_0 is 0
    elif sigma_0 == 0 :
        ketPhi = state[int(state.shape[0]/2):]/sigma_1
        return (qc.ket1, ketPhi)
    
    # And if sigma_1 is 0
    else :
        ketChi = state[:int(state.shape[0]/2)]/sigma_0
        return (qc.ket0, ketChi)


def last(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the last qbit. Returns a pair consisting of an (n - 1)-qbit state and a classical 1-qbit state (either |0> or |1>).'''
    sigma_0 = 0
    sigma_1 = 0
    for i in range(0, int(state.shape[0]), 2) :
        sigma_0 += abs(state[i])**2
    
    for j in range(1, int(state.shape[0]), 2) :
        sigma_1 += abs(state[j])**2
    
    sigma_0 = numpy.sqrt(sigma_0)
    sigma_1 = numpy.sqrt(sigma_1)

    if (sigma_0 != 0 and sigma_1 != 0) :
        ketChi = numpy.array([state[i] for i in range(0, int(state.shape[0]), 2)])/sigma_0
        ketPhi = numpy.array([state[j] for j in range(1, int(state.shape[0]), 2)])/sigma_1
        states = [(ketChi, qc.ket0), (ketPhi, qc.ket1)]
        weights = [abs(sigma_0)**2, abs(sigma_1)**2]

        results = random.choices(states, weights)
        return results[0]
    
    elif sigma_0 == 0 :
        ketPhi = numpy.array([state[j] for j in range(1, int(state.shape[0]), 2)])/sigma_1
        return (ketPhi, qc.ket1)
    
    else:
        ketChi = numpy.array([state[i] for i in range(0, int(state.shape[0]), 2)])/sigma_0
        return (ketChi, qc.ket0)



### DEFINING SOME TESTS ###

def firstTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |0> |psi> or |1> |psi>, measures the first qbit, and then reconstructs the state.
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket0, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest first part")
    else:
        print("failed firstTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket1, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest second part")
    else:
        print("failed firstTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def firstTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(qc.ket0, ketChi) + psi1 * qg.tensor(qc.ket1, ketPhi)
    def f():
        if (first(ketOmega)[0] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check firstTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))

def lastTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |psi> |0> or |psi> |1>, measures the last qbit, and then reconstructs the state.
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket0)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest first part")
    else:
        print("failed lastTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket1)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest second part")
    else:
        print("failed lastTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def lastTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(ketChi, qc.ket0) + psi1 * qg.tensor(ketPhi, qc.ket1)
    def f():
        if (last(ketOmega)[1] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check lastTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))



### RUNNING THE TESTS ###

def main():
    firstTest(2)
    firstTest345(3, 10000)
    firstTest345(2, 10000)
    lastTest(1)
    lastTest(5)
    lastTest345(10, 10000)
    lastTest345(5, 10000)

if __name__ == "__main__":
    main()

