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
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
//#include <boost/numeric/ublas/io.hpp>

#include "Matrix2D.h"



double one_function(double x, double y);
void transpose(Matrix2D<double > &t, Matrix2D<double > &m);
void MPItranspose(Matrix2D<double > &t, Matrix2D<double > &m);
std::pair<size_t,size_t> splitting(const int &size, const int &rank, const size_t &number);



// Functions implemented in FORTRAN in fst.f and called from C.
// The trailing underscore comes from a convention for symbol names, called name
// mangling: if can differ with compilers.
extern  "C"{
    void fst_(double *v, int *n, double *w, int *nn);
    void fstinv_(double *v, int *n, double *w, int *nn);
}
double
poisson(size_t n, const std::function<double(double, double)> &rhs_function, const int &size, const int &rank);

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


    Matrix2D<int> test(4,4);
    Matrix2D<int> test_t(4,4);

    int fill = rank * 8;
    for(int i = rank * 2; i < rank * 2 + 2; i++){
        for(int y = 0; y < 4;y++ ){
            test(i,y) = fill;
            fill++;
        }
    }

//    if(rank == 0){
//        for (size_t i = 0; i < 4; i++) {
//            std::cout << "[ ";
//            for (size_t j = 0; j < 4; j++) {
//                std::cout << test(i,j) << " ";
//            }
//            std::cout <<"]"<< std::endl;
//        }
//    }

    std::vector<int> trans_send_buf(8);
    int chunk = 2;
    int offset = rank *  chunk;
    for(int i = rank * 2; i < rank * 2 + 2; i++){
        for(int p = 0; p < size; p++){
            for(int y = 0; y < chunk; y++){
                trans_send_buf.at(p * 4 + (((i - offset) * chunk) + y)) = test(i, p * chunk + y);
                if(rank == 0){
                    std::cout << "(" << p << "," << (i - offset) * chunk + y << ")" << " - (" << i << "," << p * chunk + y << ") - " << p * 4 + ((i - offset) * chunk + y)  << std::endl;
                }
            }
        }

    }

    if(rank == 0){
        for (size_t i = 0; i < 8; i++) {
            std::cout << *(trans_send_buf.data() + i) << std::endl;
        }

    }

    std::vector<int> trans_recv_buf(8);
    MPI_Alltoall(trans_send_buf.data(), 4, MPI_INT, trans_recv_buf.data(), 4, MPI_INT, MPI_COMM_WORLD );

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout << "----------------------\n";
    MPI_Barrier(MPI_COMM_WORLD);

//    if(rank == 0){
//        for (size_t i = 0; i < 2; i++) {
//            std::cout << "[ ";
//            for (size_t j = 0; j < 4; j++) {
//                std::cout << trans_recv(i,j) << " ";
//            }
//            std::cout <<"]"<< std::endl;
//        }
//    }

//    if(rank == 1){
//        for (size_t i = 0; i < 4; i++) {
//            std::cout << "[ ";
//            for (size_t j = 0; j < 4; j++) {
//                std::cout << test_t(i,j) << " ";
//            }
//            std::cout <<"]"<< std::endl;
//        }
//    }


    //To the work in a separate function to make it testable
//    auto u_max = poisson(n, one_function, size, rank);
//
//
//    if(rank == 0){      //only 0 should have the full result
//        std::cout << "u_max = " << u_max <<  " vs. 0.0727826" << std::endl;
//    }

    MPI_Finalize();



    return 0;
}


/**
 * Solve the poisson problem
 * @param n The problem size (must be power of 2)
 * @param rhs_function this is the right hand size function that you be used
 * @return aximal value of solution for convergence analysis in L_\infty norm.
 */
double
poisson(const size_t n, const std::function<double(double, double)> &rhs_function, const int &size, const int &rank) {
//    using namespace boost::numeric::ublas;      //otherwise we will get very long lines...

    size_t points = n + 1;
    size_t m = n - 1;
    double h = 1.0 / n;

    // To some calculation for splitting the data
    auto points_splitting = splitting(size, rank, points);
    size_t chunk_points = points_splitting.first;
    size_t points_offset = points_splitting.second;
    auto m_slitting = splitting(size, rank, m);
    size_t chunk_m = m_slitting.first;
    size_t m_offset = m_slitting.second;


    /*
     * Grid points are generated with constant mesh size on both x- and y-axis.
     */
    //We need the full grid on every node
    auto grid = std::vector<double>(chunk_points * size);
    //each node creates a part of the grid
    for (size_t i = points_offset; i < points_offset + chunk_points; i++) {
        grid.at(i) = i * h;
    }

    //each sync the computed points so each node has the full grid at the end
    MPI_Allgather(grid.data() + points_offset  , chunk_m, MPI_DOUBLE,grid.data(), chunk_m,
            MPI_DOUBLE, MPI_COMM_WORLD);


    /*
     * The diagonal of the eigenvalue matrix of T is set with the eigenvalues
     * defined Chapter 9. page 93 of the Lecture Notes.
     * Note that the indexing starts from zero here, thus i+1.
     */
    auto diag = std::vector<double>(chunk_m * size);     //again each node need the full vector so get the full memory on each node

    for(size_t i = m_offset; i < m_offset + chunk_m; i++){
        diag.at(i) = 2.0 * (1.0 - cos((i+1) * M_PI / n));
    }

    MPI_Allgather(diag.data() + m_offset , chunk_m , MPI_DOUBLE, diag.data(), chunk_m, MPI_DOUBLE,
            MPI_COMM_WORLD);


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
    auto z = std::vector<double>(nn);


    /*
     * Initialize the right hand side data for a given rhs function. We get G
     *
     */
    ///gernerate Data via MPI
    for(size_t i = m_offset; i < std::min(m_offset + chunk_m, m); i++){
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
    for(size_t i = m_offset; i < std::min(m_offset + chunk_m, m); i++){
            fst_(b.row_ptr(i) , &n_int, z.data(), &nn_int);
    }

    MPI_Allgather(b.row_ptr(m_offset) , chunk_m  * m, MPI_DOUBLE, b.base_ptr(), chunk_m * m, MPI_DOUBLE, MPI_COMM_WORLD);





    transpose(bt, b);
    /// Do the transpose via MPI

    ///
    for (size_t i = 0; i < m; i++) {
        fstinv_(bt.row_ptr(i), &n_int, z.data(), &nn_int);
    }

    // bt = \tilde G ^T



    // Step 2
    /*
     * Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
     */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt(i,j) = bt(i,j) / (diag.at(i) + diag.at(j));
        }
    }


    //step 3
    /*
     * Compute U = S^-1 * (S * U \tilde^T) (Chapter 9. page 101 step 3)
     */
    for (size_t i = 0; i < m; i++) {
        fst_(bt.row_ptr(i), &n_int, z.data(), &nn_int);
    }
    transpose(b, bt);
    /// Do the transpose via MPI

    ///
    for (size_t i = 0; i < m; i++) {
        fstinv_(b.row_ptr(i), &n_int, z.data(), &nn_int);
    }


    /*
     * Compute maximal value of solution for convergence analysis in L_\infty
     * norm.
     */
    double u_max = 0.0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            u_max = u_max > fabs(b(i,j)) ? u_max : fabs(b(i,j));}
    }

    ///collect final u_max on rank 0

    ///


    return u_max;
}

/**
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to switch between problem definitions.
 */
double one_function(double x, double y) {
    return 1;
}


/**
 * Transposes matrix m and saves it in given matrix t
 * @param t OUTPUT the matrix where the transposed matrix m should be stored to
 * @param m INPUT matrix to transposed
 */
void transpose(Matrix2D<double > &t, Matrix2D<double > &m){
    for (size_t i = 0; i < m.getColumns(); i++) {
        for (size_t j = 0; j < m.getRows(); j++) {
            t(i,j) = m(j,i);
        }
    }
}



/**
 * Splits a given number into chunks, for the given MPI size and rank
 * If there is a rest, the chunks will be made so big, that there is a padding for the last node
 * @param size
 * @param rank
 * @param number
 * @return
 */
std::pair<size_t,size_t> splitting(const int &size, const int &rank, const size_t &number){
    size_t chunk;
    size_t offset;
    if((size > 2) && (number % size != 0)){
        chunk = number / (size - 1);
        offset = rank * chunk ;

    } else if (size == 2 && (number % size != 0)){
        chunk = number / 2 + 1;
        offset = rank * chunk;
    } else {
        chunk = number / size;
        offset = rank * chunk;
    }

    return {chunk, offset};
}

void MPItranspose(Matrix2D<double > &t, Matrix2D<double > &m){

}