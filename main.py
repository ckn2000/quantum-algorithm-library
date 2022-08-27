
import qConstants as qc
import qBitStrings as qb
import qUtilities as qu

# It is conventional to have a main() function. Change it to do whatever you want. On Day 06 you could put your entanglement experiment in here.
def main():
    for i in range(10) :
        ketPsi = qu.uniform(2)
        print(ketPsi)
        difference = ketPsi[0]*ketPsi[3]-ketPsi[1]*ketPsi[2]

        print("0th decile: ", float("{:.0f}".format(difference.real))+float("{:.0f}".format(difference.imag))*1j)

        print("10th decile: ", float("{:.10f}".format(difference.real))+float("{:.10f}".format(difference.imag))*1j)

        print("20th decile: ", float("{:.20f}".format(difference.real))+float("{:.20f}".format(difference.imag))*1j)
        

# If the user imports this file into another program as a module, then main() does not run. But if the user runs this file directly as a program, then main() does run.
if __name__ == "__main__":
    main()