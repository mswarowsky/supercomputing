#include <iostream>
#include <cmath>
#include <mach.h>


int main(int argc, char *argv[]) {

    if(argc < 2){
        std::cout << "To less arguments, please provide a n!";
        return 1;
    }

    size_t n = std::strtoul(argv[1], nullptr, 0);

    std::cout << "Using n = " << n << std::endl;

    auto approx_pi = mach::getPIMachSingleT(n);

    std::cout << "Approximated PI: " << approx_pi << "\n"<< "error: " << fabs(approx_pi - M_PI) ;

    return 0;
}