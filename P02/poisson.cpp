/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <functional>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <fstream>

#include "Matrix2D.h"

#define PLOTRANK 0

double one_function(double x, double y);
double test_function(double y, double x);
double validate_test_function(double y, double x);
double poissionTest(Matrix2D<double> & u , std::vector<double> &grid, const size_t n);
void MPI_Transpose(Matrix2D<double > &t,Matrix2D<double > &m, size_t rank, size_t size, std::vector<int> & chunks, std::vector<int> & offsets);
std::pair<size_t,size_t> splitting(const int &size, const int &rank, const size_t &number);

// Functions implemented in FORTRAN in fst.f and called from C.
// The trailing underscore comes from a convention for symbol names, called name
// mangling: if can differ with compilers.
extern  "C"{
    void fst_(double *v, int *n, double *w, int *nn);
    void fstinv_(double *v, int *n, double *w, int *nn);
}

std::pair<double , double> poisson(size_t n, const std::function<double(double, double)> &rhs_function, const int &size, const int &rank);

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: \n poisson n\n\n"
                  << "Arguments:\n  n: the problem size (must be a power of 2)" << std::endl;
        return 1;
    }

    /*
     *  The equation is solved on a 2D structured grid and homogeneous Dirichlet
     *  conditions are applied on the boundary:
     *  - the number of grid points in each direction is n+1,
     *  - the number of degrees of freedom in each direction is m = n-1,
     *  - the mesh size is constant h = 1/n.
     */
    size_t n = std::strtoul(argv[1], nullptr, 0);

    if ((n & (n-1)) != 0) {
        std::cout << "n must be a power-of-two\n" << std::endl;
        return 2;
    }

    // Starting MPI
    int size, rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time_start;
    if (rank == 0) {
        time_start = MPI_Wtime();
    }

    //maybe the code gets a little bit tidier by a separate function...
    auto results = poisson(n, test_function , size, rank);



    if(rank == 0){      //only 0 should have the full result
        double duration = MPI_Wtime() - time_start;
//        std::cout << "u_max = " << results.first <<  " vs. 0.0727826" << std::endl;
        std::cout << "Time: " << duration << " s  max error:" <<  results.second  << "   u_max: " << results.first << std::endl;
        std::fstream outPutFile("poisson.txt", std::ios::app);
        outPutFile << size << ";" << omp_get_num_threads() << ";" << n <<  ";" << results.first << ";" << results.second << ";" << duration << "\n";
        outPutFile.close();
    }

    MPI_Finalize();
    return 0;
}

/**
 * Solve the poisson problem
 * @param n The problem size (must be power of 2)
 * @param rhs_function this is the right hand size function that you be used
 * @return aximal value of solution for convergence analysis in L_\infty norm.
 */
std::pair<double , double> poisson(const size_t n, const std::function<double(double, double)> &rhs_function, const int &size, const int &rank) {
    size_t points = n + 1;
    size_t m = n - 1;
    double h = 1.0 / n;

    auto chunks_points = std::vector<int>(size);
    auto chunks_m = std::vector<int>(size);
    auto offsets_points = std::vector<int>(size);
    auto offsets_m = std::vector<int>(size);
    std::vector<int> chunks_matrix(size);
    std::vector<int> offsets_matrix(size);


    //Could be done via MPI but probably faster if every node does it local
    for(int i = 0; i < size;i++){
        auto points_splitting = splitting(size, i, points);
        auto m_slitting = splitting(size, i, m);

        chunks_points.at(i) = points_splitting.first;
        offsets_points.at(i) = points_splitting.second;

        chunks_m.at(i) = m_slitting.first;
        offsets_m.at(i) = m_slitting.second;

        chunks_matrix.at(i) = m_slitting.first * m;
        offsets_matrix.at(i) = m_slitting.second * m;
    }

    // To some calculation for splitting the data
    size_t chunk_points = chunks_points.at(rank);
    size_t points_offset = offsets_points.at(rank);
    size_t chunk_m = chunks_m.at(rank);
    size_t m_offset = offsets_m.at(rank);



    /*
     * Grid points are generated with constant mesh size on both x- and y-axis.
     */
    //We need the full grid on every node
    auto grid = std::vector<double>(points);
    //each node creates a part of the grid
#pragma omp parallel for
    for (size_t i = points_offset; i < points_offset + chunk_points; i++) {
        grid.at(i) = i * h;
    }
    //each sync the computed points so each node has the full grid at the end
    MPI_Allgatherv(grid.data() + points_offset  , chunk_points, MPI_DOUBLE, grid.data(), chunks_points.data(), offsets_points.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    /*
     * The diagonal of the eigenvalue matrix of T is set with the eigenvalues
     * defined Chapter 9. page 93 of the Lecture Notes.
     * Note that the indexing starts from zero here, thus i+1.
     */
    auto diag = std::vector<double>(m);     //again each node need the full vector so get the full memory on each node
#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        diag.at(i) = 2.0 * (1.0 - cos((i+1) * M_PI / n));
    }
    MPI_Allgatherv(diag.data() + m_offset , chunk_m , MPI_DOUBLE, diag.data(), chunks_m.data(), offsets_m.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    /*
     * Allocate the matrices b and bt which will be used for storing value of
     * G, \tilde G^T, \tilde U^T, U as described in Chapter 9. page 101.
     */
    Matrix2D<double > b(m, m);
    Matrix2D<double> bt(m,m);

    /*
     * This vector will holds coefficients of the Discrete Sine Transform (DST)
     * but also of the Fast Fourier Transform used in the FORTRAN code.
     * The storage size is set to nn = 4 * n, look at Chapter 9. pages 98-100:
     * - Fourier coefficients are complex so storage is used for the double part
     *   and the imaginary part.
     * - Fourier coefficients are defined for j = [[ - (n-1), + (n-1) ]] while
     *   DST coefficients are defined for j [[ 0, n-1 ]].
     * As explained in the Lecture notes coefficients for positive j are stored
     * first.
     * The array is allocated once and passed as arguments to avoid doings
     * doublelocations at each function call.
     */
    size_t nn = 4 * n;
//    Now needs to be initialized privately for each OpenMP intance
//    auto z = std::vector<double>(nn);


    /*
     * Initialize the right hand side data for a given rhs function. We get G
     *
     */
#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        for (size_t j = 0; j < m; j++) {
            b(i,j) = h * h * rhs_function(grid.at(i+1), grid.at(j+1));
        }
    }


    /*
     * Step 1
     * Compute \tilde G^T = S^-1 * (S * G)^T (Chapter 9. page 101 step 1)
     * Instead of using two matrix-matrix products the Discrete Sine Transform
     * (DST) is used.
     * The DST code is implemented in FORTRAN in fst.f and can be called from C.
     * The array zz is used as storage for DST coefficients and internally for
     * FFT coefficients in fst_ and fstinv_.
     * In functions fst_ and fst_inv_ coefficients are written back to the input
     * array (first argument) so that the initial values are overwritten.
     */
    int n_int = static_cast<int>(n);    // int cast for fortan code .... :(
    int nn_int = static_cast<int>(nn);
#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        auto zz = std::vector<double>(nn);
        fst_(b.row_ptr(i) , &n_int, zz.data(), &nn_int);
    }

    MPI_Transpose(bt, b, (size_t) rank, (size_t) size, chunks_m, offsets_m);

#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        auto zz = std::vector<double>(nn);
        fstinv_(bt.row_ptr(i), &n_int, zz.data(), &nn_int);
    }


    // Step 2
    /*
     * Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
     */
#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        for (size_t j = 0; j < m; j++) {
            bt(i,j) = bt(i,j) / (diag.at(i) + diag.at(j));
        }
    }


    //step 3
    /*
     * Compute U = S^-1 * (S * U \tilde^T) (Chapter 9. page 101 step 3)
     */
#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        auto zz = std::vector<double>(nn);
        fst_(bt.row_ptr(i), &n_int, zz.data(), &nn_int);
    }

    MPI_Transpose(b, bt, (size_t) rank, (size_t) size, chunks_m, offsets_m);

#pragma omp parallel for
    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        auto zz = std::vector<double>(nn);
        fstinv_(b.row_ptr(i), &n_int, zz.data(), &nn_int);
    }


    //get the matrix to all nodes alternatively just send it to rank 0 but should also not that much slower this way
    MPI_Allgatherv(b.row_ptr(m_offset) , chunk_m  * m, MPI_DOUBLE, b.base_ptr(), chunks_matrix.data(), offsets_matrix.data(), MPI_DOUBLE, MPI_COMM_WORLD);


    auto max_error = poissionTest(b, grid, n);


    /*
     * Compute maximal value of solution for convergence analysis in L_\infty
     * norm.
     */
    double u_max = 0.0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            u_max = u_max > fabs(b(i,j)) ? u_max : fabs(b(i,j));}
    }
    return {u_max, max_error};
}

/**
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to switch between problem definitions.
 */
double one_function(double x, double y) {
    return 1;
}

/**
 * Recommended test function from Appendix B.
 * With this we test the implementation on correctness
 */
double test_function(double y, double x){
    return 5 * M_PI * M_PI * validate_test_function(y,x);
}

/**
 *  The validation function for the test function
 */
double validate_test_function(double y, double x) {
    return sin(M_PI * x) * sin(2 * M_PI * y);
}

/**
 * Unit test for the caculation
 */
double poissionTest(Matrix2D<double> & u , std::vector<double> &grid, const size_t n){
    size_t dimention = n - 1;
    double max_error = 0.0;


    for(int y = 0; y < dimention; y++ ){
        for (int x = 0; x < dimention; ++x) {
            if(fabs(u(y,x) -  validate_test_function(grid[y + 1], grid[x + 1])) > max_error) {
                max_error = fabs(u(y,x) -  validate_test_function(grid[y + 1], grid[x + 1]));
            }

        }
    }
    return max_error;
}

/**
 * Splits a given number into chunks, for the given MPI size and rank
 * If there is a rest, the chunks will be made so big, that there is a padding for the last node
 * @param size
 * @param rank
 * @param number
 * @return {chunk, offset}
 */
std::pair<size_t,size_t> splitting(const int &size, const int &rank, const size_t &number){
    size_t chunk;
    size_t offset;
    if((size > 2) && (number % size != 0)){
        chunk = number / (size - 1);

    } else if (size == 2 && (number % size != 0)){
        chunk = number / 2 + 1;
    } else {
        chunk = number / size;
    }

    offset = rank * chunk;
    //some more special stuff for the last node
    if(rank == size -1 && offset + chunk > number){
        chunk = number - offset;
    }
    return {chunk, offset};
}

void MPI_Transpose(Matrix2D<double > &t,Matrix2D<double > &m, size_t rank, size_t size, std::vector<int > & chunks, std::vector<int> & offsets){
    auto chunk = chunks.at(rank);
    auto offset = offsets.at(rank);

    auto dim = m.getColumns();
    std::vector<int> data_per_node(size);
    std::vector<int> offsets_per_node(size);
    offsets_per_node.at(0) = 0;
    for(int i = 0; i < data_per_node.size(); i++){
        data_per_node.at(i) = chunk * chunks.at(i);
        offsets_per_node.at(i) = std::accumulate(data_per_node.begin(), data_per_node.begin() + i, 0);
    }


    std::vector<double > trans_send_buf(dim * chunk);
    for(size_t i = offset; i < offset + chunk; i++){
        for(int p = 0; p < size; p++){
            auto split_chunk = chunks[p];
            for(int y = 0; y < split_chunk; y++) {
                trans_send_buf.at(std::accumulate(data_per_node.begin(), data_per_node.begin() + p, 0) +
                                  (i - offset) * split_chunk + y) = m(i, std::accumulate(chunks.begin(), chunks.begin() + p, 0) + y);
            }
        }
    }

    std::vector<double > trans_recv_buf(dim * chunk);
    MPI_Alltoallv(trans_send_buf.data(), data_per_node.data(), offsets_per_node.data(), MPI_DOUBLE,
            trans_recv_buf.data(), data_per_node.data(), offsets_per_node.data() , MPI_DOUBLE, MPI_COMM_WORLD );

    for(size_t d = 0; d < trans_recv_buf.size(); d++){
        //calculate row and column extra compile will optimize that ....
        size_t row = offset + d % chunk;
        size_t column = d / chunk;
        t(row, column) = trans_recv_buf.at(d);
    }
}