#include <iostream>
#include <cmath>
#include <mach.h>
#include <chrono>
#include <omp.h>
#include <fstream>


int main(int argc, char *argv[]) {

    if(argc < 2){
        std::cout << "To less arguments, please provide a n!";
        return 1;
    }

    size_t n = std::strtoul(argv[1], nullptr, 0);
    std::cout << "Using n = " << n << std::endl;

    auto start = std::chrono::system_clock::now();

    double x = 1./5.;
    double y = 1/239.;

    double series_1 = 0.0;
    double series_2 = 0.0;

    //static schedule as we expect equal runtime for each element
#pragma omp parallel for reduction(+:series_1) reduction(+:series_2) schedule(static)
    for(long i = 1; i <= n; i++){
        series_1 += mach::machElement(i, x);
        series_2 += mach::machElement(i, y);
    }


    auto approx_pi = mach::getPIFromArctans(series_1, series_2);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now() - start);
    std::cout << "pi: " << approx_pi << ", "<< "error: " << fabs(approx_pi - M_PI) << ", duration: " << duration.count() << "ms"
              << std::endl ;
    std::fstream outPutFile("mach3.txt", std::ios::app);
    outPutFile << omp_get_num_threads() << ";" << n << ";"<< approx_pi << ";" << fabs(approx_pi - M_PI) << ";" << duration.count() << "\n";
    outPutFile.close();

    return 0;
}