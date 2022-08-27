


# Let's talk about how the CS 358 Python project is handed in and graded.

# If you have conferred with anyone else, then make sure everyone responsible for the work is properly attributed in the code. Depending on how frequently you collaborate, you might have to attribute function-by-function.

# Try to make the code as clean and understandable as possible, without doing a major rewrite that might damage it. Ideally, code uses meaningful identifiers and metaphors, so that it is self-documenting. But this ideal is rarely achieved. Consider inserting comments to explain anything that isn't clear.

# If your code makes a bunch of unasked-for print statements, then please comment them out. For example, a test function should probably print out a message about passing or failing, but it should not print out a bunch of extra diagnostic information left over from your debugging.

# If you know that some part of your project isn't working up to specification, then type up a plain text file readme.txt. In it, briefly explain everything that's broken. If you do not tell me what's broken, then I infer that you think that everything is working.

# Even if your code is all working, you are welcome to include a readme.txt, if for some reason you wish to frame how I read your code.

# By my count, our simulator library consists of six files:
#    qAlgorithms.py
#    qBitStrings.py
#    qConstants.py
#    qGates.py
#    qMeasurement.py
#    qUtilities.py
# Put these six files into a folder. If you have helper files, then include them. If you have a readme.txt, then include it. Please don't include anything unnecessary. Make a ZIP archive of the folder and e-mail it to me by 5:00 PM on Wednesday March 16.

# While grading your project, I run several tests on it. Most of the tests are ones that you already have, because I've given them to you or asked you to write them. But I might insert a couple of other tests. I haven't decided yet. See below for two simple examples. (If you delete 'Sol' from certain file names below, then you can run this test file yourself.)

# While the tests run, I read your readme.txt if any. Then I skim your code, to see whether it makes sense. (This is not a software engineering course, so I do not have high standards for your code quality. But if I cannot understand your code, even though I specified it and wrote similar code myself, then that's not great.) Then I examine the test results.



import random
import numpy

# Import the student's library.
import qConstants as qc
import qUtilities as qu
import qBitStrings as qb
import qGates as qg
import qMeasurement as qm
import qAlgorithms as qa

# Import my library. qBitStringsSol should be identical to qBitStrings.
import qConstants as qcs
import qUtilities as qus
import qBitStrings as qbs
import qGates as qgs
import qMeasurement as qms
import qAlgorithms as qas

def simonTestSimple(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Let k be the index of the first 1 in delta.
    k = 0
    while delta[k] == 0:
        k += 1
    # This matrix M is always its own inverse mod 2.
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it's a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    # Check whether simon outputs a bit string perpendicular to delta.
    kets = qa.simon(n, gate)
    bits = tuple(map(qus.bitValue, kets))
    if qbs.dot(bits, delta) == 0:
        print("passed simonTestSimple")
    else:
        print("failed simonTestSimple")
        print("    delta = " + str(delta))
        print("    bits = " + str(bits))

def gateTestSimple():
    first = numpy.matmul(qc.swap, numpy.matmul(qc.cnot, qc.swap))
    hh = qg.tensor(qc.h, qc.h)
    second = numpy.matmul(hh, numpy.matmul(qc.cnot, hh))
    if qus.equal(first, second, 0.000001):
        print("passed gateTestSimple")
    else:
        print("failed gateTestSimple")
        print("    first = " + str(first))
        print("    second = " + str(second))

def main():
    try:
        gateTestSimple()
    except:
        print("failed gateTestSimple")
        print("    fatal error")
    try:
        simonTestSimple(5)
    except:
        print("failed simonTestSimple")
        print("    fatal error")
    try:
        qa.shorTest(5, 5)
    except:
        print("failed shorTest")
        print("    fatal error")

if __name__ == "__main__":
    main()


