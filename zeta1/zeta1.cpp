/**
 * An MPI implementation where MPI is used to sum a vector
 */


#include <zeta.h>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <numeric>

int main(int argc, char *argv[]) {

    if(argc < 2){
        std::cout << "To less arguments, please provide a n!";
        return 1;
    }


    // Starting MPI
    int size, rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



    size_t n = std::strtoul(argv[1], nullptr, 0);
    if(rank == 0) {
        std::cout << "Using n = " << n << std::endl;
    }



    // To some calculation for splitting the data
    size_t chunk;
    if((size > 1) && (n % size != 0)){
        chunk = n / (size - 1);
    } else {
        chunk = n / size;
    }


    // compute the vector elements
    std::vector<double> v;
    v.resize(chunk * size);   //to allocate already all memory needed

    if(rank == 0){
        for(size_t i = 1; i < n; i++) {
            v.at(i) = zeta::zetaElement(i);
        }
    }


    if(rank == 0){
        std::cout << "Sending elements to " << size << " other processes.\n";
    }
    //Splitting Data and sending it to all the processes
    MPI_Bcast(v.data(), static_cast<int >(n), MPI_DOUBLE, 0 , MPI_COMM_WORLD);
    
    //sum up the scattered elements
    double pi = std::accumulate(v.begin(), v.end(), 0.0);
    pi = zeta::getPIfromZetaSeries(pi);
    
    if(rank == 0){
        std::cout << "Approximated PI: " << pi << "\n"<< "error: " << fabs(pi - M_PI) ;
    }

    MPI_Finalize();
    return 0;
}