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



    size_t n = strtoul(argv[1], nullptr, 0);
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
    std::vector<double> part_v;
    part_v.resize(chunk);
    MPI_Scatter(v.data(), static_cast<int>(chunk), MPI_DOUBLE, part_v.data(), static_cast<int>(chunk), MPI_DOUBLE, 0, MPI_COMM_WORLD);


    //sum up the scattered elements
    double local_pi = std::accumulate(part_v.begin(), part_v.end(), 0.0);


    double global_pi;
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    global_pi = zeta::getPIfromZetaSeries(global_pi);



    if(rank == 0){
        std::cout << "Approximated PI: " << global_pi << "\n"<< "error: " << fabs(global_pi - M_PI) ;
    }

    MPI_Finalize();
    return 0;
}