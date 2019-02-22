/**
 * An MPI implementation where MPI is used a little bit smarter
 */


#include <zeta.h>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <cassert>
#include <fstream>

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

    //we want power of 2 nodes
    if((size & (size - 1)) != 0) {
        std::cout << "Number of nodes must be a power of two\n";
        return 1;
    }

    double time_start;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }



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




    if(rank == 0){
        std::cout << "Sending elements to " << size << " other processes.\n";
    }
    //computing the dat on each node add creating the partial sum
    std::vector<double> local_v;
    local_v.resize(chunk);

    //starting the series at 1 but the index is starting at 0, so adding +1 in for the zeta element
    size_t offset = chunk * rank + 1;
    for(size_t i = 0; i < chunk; i++) {
        local_v.at(i) = zeta::zetaElement(i + offset);
    }

    //sum up the scattered elements
    double local_pi = std::accumulate(local_v.begin(), local_v.end(), 0.0);



    //Collecting the data on the root(0) node
    double global_pi;
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //getting PI and printing it
    global_pi = zeta::getPIfromZetaSeries(global_pi);
    if(rank == 0){
        double duration = MPI_Wtime() - time_start;
        std::cout << "pi: " << global_pi << ", "<< "error: " << fabs(global_pi - M_PI) << ", duration: " << duration
                  << std::endl ;
        std::fstream outPutFile("zeta2.txt", std::ios::app);
        outPutFile << size << ";" << global_pi << ";" << fabs(global_pi - M_PI) << ";" << duration << "\n";
        outPutFile.close();
    }

    MPI_Finalize();
    return 0;
}