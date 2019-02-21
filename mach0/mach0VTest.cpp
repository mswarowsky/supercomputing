/**
 * Verification test for machin
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <mach.h>

#define TEST_RUNS 24

int main() {
    std::ofstream outPutFile;
    outPutFile.open("mach0VTest.txt");
    for(long i = 1; i <= TEST_RUNS; i++){
        long n = (0x1 << i); //testing with n = 2^i
        auto pi = mach::getPIMachSingleT(n);
        std::cout << "error PI - PI_" << n << " : " << fabs(M_PI - pi) << "\n";
        outPutFile << n << ";" << fabs(M_PI - pi) << "\n";
    }

    outPutFile.close();

    return 0;
}