/**
 * Verification test for zeta
 */

#include <zeta.h>
#include <iostream>
#include <cmath>

#define TEST_RUNS 24

int main() {
    for(long i = 1; i <= TEST_RUNS; i++){
        long n = (0x1 << i); //testing with n = 2^i
        auto pi = zeta::getPIfromZetaSeries(zeta::singleZeta(n));
        std::cout << "error PI - PI_" << n << " : " << fabs(M_PI - pi) << "\n";
    }

    return 0;
}
