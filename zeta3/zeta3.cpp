#include <iostream>
#include <fstream>
#include <zeta.h>
#include <cmath>
#include <omp.h>
#include <chrono>


int main(int argc, char *argv[]) {

    if(argc < 2){
        std::cout << "To less arguments, please provide a n!";
        return 1;
    }

    size_t n = std::strtoul(argv[1], nullptr, 0);
    std::cout << "Using n = " << n << std::endl;

    auto start = std::chrono::system_clock::now();

    double approx_pi = 0.0;

#pragma omp parallel for reduction(+:approx_pi) schedule(static)
    for(long i = 1; i <= n; i++){
        approx_pi += zeta::zetaElement(i);
    }

    approx_pi = zeta::getPIfromZetaSeries(approx_pi);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start);
    std::cout << "pi: " << approx_pi << ", "<< "error: " << fabs(approx_pi - M_PI) << ", duration: " << duration.count() << "us"
            << std::endl ;
    std::fstream outPutFile("zeta3.txt", std::ios::app);
    outPutFile << omp_get_num_threads() << ";" << n << ";"<< approx_pi << ";" << fabs(approx_pi - M_PI) << ";" << duration.count() << "\n";
    outPutFile.close();

    return 0;
}