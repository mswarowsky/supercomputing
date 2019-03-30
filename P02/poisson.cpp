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
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
//#include <boost/numeric/ublas/io.hpp>

#include "Matrix2D.h"



double one_function(double x, double y);
void transpose(Matrix2D<double > &t, Matrix2D<double > &m);
void MPI_Transpose(Matrix2D<double > &t,Matrix2D<double > &m, size_t rank, size_t size, std::vector<int> & chunks, std::vector<int> & offsets);
std::pair<size_t,size_t> splitting(const int &size, const int &rank, const size_t &number);

#define PLOTRANK 0

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

    size_t test_dim = 5;
    size_t chunk = splitting(size, rank, test_dim).first;
    size_t offset = splitting(size, rank, test_dim).second;

    Matrix2D<double> test(test_dim,test_dim);
    Matrix2D<double> test_T(test_dim,test_dim);

    std::vector<int> chunks({3,2});
    std::vector<int> offsets({0,3});


    double fill = rank * (test_dim*test_dim/size);
    for(int i = offset; i < offset + chunk; i++){
        for(int y = 0; y < test_dim;y++ ){
            test(i,y) = fill;
            fill++;
        }
    }

    if(rank == PLOTRANK){
        for (size_t i = 0; i < test_dim; i++) {
            std::cout << "[ ";
            for (size_t j = 0; j < test_dim; j++) {
                std::cout << test(i,j) << " ";
            }
            std::cout <<"]"<< std::endl;
        }
    }

    MPI_Transpose(test_T, test, rank, size, chunks, offsets);


    if(rank == PLOTRANK){
        for (size_t i = 0; i < test_dim; i++) {
            std::cout << "[ ";
            for (size_t j = 0; j < test_dim; j++) {
                std::cout << test_T(i,j) << " ";
            }
            std::cout <<"]"<< std::endl;
        }
    }

    //To the work in a separate function to make it testable
    auto u_max = poisson(n, one_function, size, rank);


    if(rank == 0){      //only 0 should have the full result
        std::cout << "u_max = " << u_max <<  " vs. 0.0727826" << std::endl;
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
double
poisson(const size_t n, const std::function<double(double, double)> &rhs_function, const int &size, const int &rank) {
//    using namespace boost::numeric::ublas;      //otherwise we will get very long lines...

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
    auto z = std::vector<double>(nn);


    /*
     * Initialize the right hand side data for a given rhs function. We get G
     *
     */
    for(size_t i = m_offset; i < m; i++){
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


    MPI_Allgatherv(b.row_ptr(m_offset) , chunk_m  * m, MPI_DOUBLE, b.base_ptr(), chunks_matrix.data(), offsets_matrix.data(), MPI_DOUBLE, MPI_COMM_WORLD);



    std::cout << "r:" << rank << " ";
    for (auto d: diag) {
        std::cout << d << ", ";
    }
    std::cout << "]" << std::endl;

    transpose(bt, b);
//    MPI_Transpose(bt, b, (size_t) rank, (size_t) size, chunk_m, m_offset);
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



    std::cout << "Rank:" << rank << " chunk:" << chunk << " offset:" << offset <<  " of " << number << std::endl;

    return {chunk, offset};
}

void MPI_Transpose(Matrix2D<double > &t,Matrix2D<double > &m, size_t rank, size_t size, std::vector<int > & chunks, std::vector<int> & offsets){

    auto chunk = chunks.at(rank);
    auto offset = offsets.at(rank);

    auto dim = m.getColumns();
    std::vector<int> data_per_node(size);
    for(int i = 0; i < data_per_node.size(); i++){
        data_per_node.at(i) = chunk * chunks.at(i);
    }


    std::vector<double > trans_send_buf(dim * chunk);
    for(size_t i = offset; i < offset + chunk; i++){
        for(int p = 0; p < size; p++){
            if(p == size -1) {  //spezial stuff for last node
                int last_chunk = chunks.back();
                for(int y = 0; y < last_chunk; y++){
//                    std::cout << "(" << i << "," << p * chunk + y << ")=>" << std::accumulate(data_per_node.begin(), data_per_node.begin() + p, 0) + (i - offset) * chunk + y << std::endl;
                    trans_send_buf.at(std::accumulate(data_per_node.begin(), data_per_node.begin() + p, 0) + (i - offset) * last_chunk + y) = m(i, p * chunk + y);
                }
            } else {
                for(int y = 0; y < chunk; y++){
//                    std::cout << "(" << i << "," << p * chunk + y << ")=>" << std::accumulate(data_per_node.begin(), data_per_node.begin() + p, 0) + (i - offset) * chunk + y << std::endl;
                    trans_send_buf.at(std::accumulate(data_per_node.begin(), data_per_node.begin() + p, 0) + (i - offset) * chunk + y) = m(i, p * chunk + y);
                }
            }
        }
    }

    if(rank == PLOTRANK) {
        std::cout << "send: [";
        for (auto d: trans_send_buf) {
            std::cout << d << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::vector<double > trans_recv_buf(dim * chunk);
//    MPI_Alltoall(trans_send_buf.data(), static_cast<int>(data_per_node), MPI_DOUBLE,
//            trans_recv_buf.data(), static_cast<int>(data_per_node), MPI_DOUBLE, MPI_COMM_WORLD );

//
//    if(rank == PLOTRANK) {
//        std::cout << "rescv: [";
//        for (auto d: trans_recv_buf) {
//            std::cout << d << ", ";
//        }
//        std::cout << "]" << std::endl;
//    }
//
//    for(size_t d = 0; d < trans_recv_buf.size(); d++){
//        //calculate row and column extra compile will optimize that ....
//        size_t row = offset + d % chunk;
//        size_t column = d / chunk;
//        t(row, column) = trans_recv_buf.at(d);
//    }

}