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

    for(size_t i = 1; i < n; i++) {
        v.emplace_back(i);
    }


    //share the elements use process 0 to share and collect all elements
    int chunk = static_cast<int>(n / size); // number of elements for each process
    std::vector<double> part_v;
    part_v.reserve(static_cast<size_t>(chunk));

    if(rank == 0){
        std::cout << "Sending elements to " << size << " other processes.\n";
    }
    //TODO handle the case that the n is not a multiple of size
    MPI_Scatter(v.data(), chunk, MPI_DOUBLE, part_v.data(), chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //TODO compute zeta partial sum here

    // gather and reduce the elements summed together
    double approx_pi;
    MPI_Reduce(part_v.data(), &approx_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if(rank == 0){
        std::cout << "Approximated PI: " << approx_pi << "\n"<< "error: " << fabs(approx_pi - M_PI) ;
    }

    MPI_Finalize();
    return 0;
}