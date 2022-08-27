import math
import numpy

# In numpy, I think that the default complex dtype varies from platform to platform. If you want to explicitly use the default type in your code, use 'one.dtype' (where one is defined just below).

# A 0-qbit state or gate is the complex scalar 1, represented as the following object. Notice that this object is neither the vector numpy.array([1 + 0j]) nor the matrix numpy.array([[1 + 0j]]).
one = numpy.array(1 + 0j)

# Our favorite one-qbit states.
ket0 = numpy.array([1 + 0j, 0 + 0j])
ket1 = numpy.array([0 + 0j, 1 + 0j])
ketPlus = numpy.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
ketMinus = numpy.array([1 / math.sqrt(2), -1 / math.sqrt(2)])

# Our favorite one-qbit gates.
i = numpy.array([
    [1 + 0j, 0 + 0j], 
    [0 + 0j, 1 + 0j]])
x = numpy.array([
    [0 + 0j, 1 + 0j], 
    [1 + 0j, 0 + 0j]])
y = numpy.array([
    [0 + 0j, 0 - 1j], 
    [0 + 1j, 0 + 0j]])
z = numpy.array([
    [1 + 0j, 0 + 0j], 
    [0 + 0j, -1 + 0j]])
h = numpy.array([
    [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j], 
    [1 / math.sqrt(2) + 0j, -1 / math.sqrt(2) + 0j]])

# Our favourite two-qbit gates.
swap = numpy.array([
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j]
])
cnot = numpy.array([
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
])