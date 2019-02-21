/**
 * An MPI implementation where MPI is used to sum a vector
 */


#include <zeta.h>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[]) {

    if(argc < 2){
        std::cout << "To less arguments, please provide a n!";
        return 1;
    }

    size_t n = std::strtoul(argv[1], nullptr, 0);
    std::cout << "Using n = " << n << std::endl;

    // Starting MPI
    int size, rank;
    MPI_Status status;
    MPI_Init(&argc, & argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_size(MPI_COMM_WORLD, &rank);



    // compute the vector elements
    std::vector<double> v;
    v.reserve(n);   //to allocate already all memory needed

    if(rank == 0){
        for(size_t i = 1; i < n; i++) {
            v.emplace_back(zeta::zetaElement(i));
        }
    }


    if(rank == 0){
        std::cout << "Sending elements to " << size << " other processes.\n";
    }
    //TODO Broadcast
    MPI_Bcast(v.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //TODO summ the brodcasted elements up ?
    double approx_pi;


    if(rank == 0){
        std::cout << "Approximated PI: " << approx_pi << "\n"<< "error: " << fabs(approx_pi - M_PI) ;
    }

    MPI_Finalize();
    return 0;
}