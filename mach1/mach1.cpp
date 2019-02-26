/**
 * An MPI implementation where MPI is used to sum a vector
 */


#include <mach.h>
#include <iostream>
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
    std::vector<double> v_1, v_2;
    v_1.resize(chunk * size);   //to allocate already all memory needed
    v_2.resize(chunk * size);



    if(rank == 0){
        // We need arctan form two values
        double x = 1./5.0;
        double y = 1./239.;

        for(size_t i = 1; i < n; i++) {
            v_1.at(i) = mach::machElement(i, x);
            v_2.at(i) = mach::machElement(i, y);
        }
    }


    if(rank == 0){
        std::cout << "Sending elements to " << size << " other processes.\n";
    }
    MPI_Bcast(v_1.data(), static_cast<int>(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v_2.data(), static_cast<int>(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    //sum up the scattered elements
    double arctan[2];
    arctan[0] = std::accumulate(v_1.begin(), v_1.end(), 0.0);
    arctan[1] = std::accumulate(v_2.begin(), v_2.end(), 0.0);

    auto global_pi = mach::getPIFromArctans(arctan[0],arctan[1]);


    if(rank == 0){
        std::cout << "Approximated PI: " << global_pi << "\n"<< "error: " << fabs(global_pi - M_PI) ;
    }

    MPI_Finalize();
    return 0;
}