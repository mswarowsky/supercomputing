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
    //Splitting Data and sending it to all the processes
    std::vector<double> part_v_1, part_v_2;
    part_v_1.resize(chunk);
    part_v_2.resize(chunk);
    MPI_Scatter(v_1.data(), static_cast<int>(chunk), MPI_DOUBLE, part_v_1.data(), static_cast<int>(chunk), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(v_2.data(), static_cast<int>(chunk), MPI_DOUBLE, part_v_2.data(), static_cast<int>(chunk), MPI_DOUBLE, 0, MPI_COMM_WORLD);



    //sum up the scattered elements
    double local_arctan[2];
    local_arctan[0] = std::accumulate(part_v_1.begin(), part_v_1.end(), 0.0);
    local_arctan[1] = std::accumulate(part_v_2.begin(), part_v_2.end(), 0.0);

    double global_arctan[2];
    MPI_Reduce(&local_arctan, &global_arctan, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_pi = mach::getPIFromArctans(global_arctan[0],global_arctan[1]);


    if(rank == 0){
        std::cout << "Approximated PI: " << global_pi << "\n"<< "error: " << fabs(global_pi - M_PI) ;
    }

    MPI_Finalize();
    return 0;
}