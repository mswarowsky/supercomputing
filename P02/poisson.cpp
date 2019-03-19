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
#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#define PI 3.14159265358979323846

typedef std::vector<std::vector<double>> matrix;

// Function prototypes
double *mk_1D_array(size_t n, bool zero);
double **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose_c(double **bt, double **b, size_t m);

double one_function(double x, double y);
matrix create_matrix(size_t rows, size_t columns);
matrix transpose(const matrix &m);
void transpose(matrix t, const matrix &m);



// Functions implemented in FORTRAN in fst.f and called from C.
// The trailing underscore comes from a convention for symbol names, called name
// mangling: if can differ with compilers.
extern  "C"{
    void fst_(double *v, int *n, double *w, int *nn);
    void fstinv_(double *v, int *n, double *w, int *nn);
}
double poisson(size_t n, const std::function<double(double, double)> &);

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

    //To the work in a separate function to make it testable
    auto u_max = poisson(n, one_function);

    std::cout << "u_max = " << u_max << std::endl;


    return 0;
}


/**
 * Solve the poisson problem
 * @param n The problem size (must be power of 2)
 * @param rhs_function this is the right hand size function that you be used
 * @return aximal value of solution for convergence analysis in L_\infty norm.
 */
double poisson(const size_t n, const std::function<double(double, double)> &rhs_function){
    using namespace boost::numeric::ublas;      //otherwise we will get very long lines...

    size_t m = n - 1;
    double h = 1.0 / n;

    /*
     * Grid points are generated with constant mesh size on both x- and y-axis.
     */
    double *grid_c = mk_1D_array(n+1, false);
    for (size_t i = 0; i < n+1; i++) {
        grid_c[i] = i * h;
    }
    auto grid = std::vector<double>(n+1);
    for (size_t i = 0; i < grid.size(); i++) {
        grid.at(i) = i * h;
    }

    /*
     * The diagonal of the eigenvalue matrix of T is set with the eigenvalues
     * defined Chapter 9. page 93 of the Lecture Notes.
     * Note that the indexing starts from zero here, thus i+1.
     */
    double *diag_c = mk_1D_array(m, false);
    for (size_t i = 0; i < m; i++) {
        diag_c[i] = 2.0 * (1.0 - cos((i+1) * PI / n));
    }
    auto diag = std::vector<double>(m);
    for (size_t i = 0; i < diag.size(); i++) {
        diag.at(i) = 2.0 * (1.0 - cos((i+1) * PI / n));
    }

    /*
     * Allocate the matrices b and bt which will be used for storing value of
     * G, \tilde G^T, \tilde U^T, U as described in Chapter 9. page 101.
     */
    double **b_c = mk_2D_array(m, m, false);
    double **bt_c = mk_2D_array(m, m, false);
//    matrix<double> b(m,m);
    auto b = create_matrix(m, m);
//    matrix<double> bt(m,m);

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
    double *z_c = mk_1D_array(nn, false);
    auto z = std::vector<double>(nn);


    /*
     * Initialize the right hand side data for a given rhs function.
     *
     */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            b.at(i).at(j) =  h * h * rhs_function(grid.at(i+1), grid.at(j+1));
            b_c[i][j] = h * h * rhs_function(grid_c[i+1], grid_c[j+1]);
        }
    }


    /*
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
    for (size_t i = 0; i < m; i++) {
        fst_(b_c[i], &n_int, z_c, &nn_int);
        fst_(b.at(i).data() , &n_int, z.data(), &nn_int);
    }
    transpose_c(bt_c, b_c, m);
    auto bt = transpose(b);
    for (size_t i = 0; i < m; i++) {
        fstinv_(bt_c[i], &n_int, z_c, &nn_int);
        fstinv_(bt.at(i).data(), &n_int, z.data(), &nn_int);
    }

    /*
     * Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
     */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt_c[i][j] = bt_c[i][j] / (diag_c[i] + diag_c[j]);
            bt.at(i).at(j) = bt.at(i).at(j) / (diag.at(i) + diag.at(j));
        }
    }

    /*
     * Compute U = S^-1 * (S * U \tilde^T) (Chapter 9. page 101 step 3)
     */
    for (size_t i = 0; i < m; i++) {
        fst_(bt_c[i], &n_int, z_c, &nn_int);
        fst_(bt.at(i).data(), &n_int, z.data(), &nn_int);
    }
    transpose_c(b_c, bt_c, m);
    transpose(b, bt);
    for (size_t i = 0; i < m; i++) {
        fstinv_(b_c[i], &n_int, z_c, &nn_int);
        fstinv_(b.at(i).data(), &n_int, z.data(), &nn_int);
    }

    /*
     * Compute maximal value of solution for convergence analysis in L_\infty
     * norm.
     */
    double u_max_c = 0.0;
    double u_max = 0.0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            u_max_c = u_max_c > fabs(b_c[i][j]) ? u_max_c : fabs(b_c[i][j]);
            u_max = u_max > fabs(b.at(i).at(j)) ? u_max : fabs(b.at(i).at(j));}
    }


    std::cout << "c++ u_max " << u_max << std::endl;

    return u_max_c;
}

/**
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to switch between problem definitions.
 */
double one_function(double x, double y) {
    return 1;
}

/**
 * Write the transpose of b a matrix of R^(m*m) in bt.
 * In parallel the function MPI_Alltoallv is used to map directly the entries
 * stored in the array to the block structure, using displacement arrays.
 */
void transpose_c(double **bt, double **b, size_t m)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = b[j][i];
        }
    }
}

/**
 * The allocation of a vectore of size n is done with just allocating an array.
 * The only thing to notice here is the use of calloc to zero the array.
 */
double *mk_1D_array(size_t n, bool zero)
{
    if (zero) {
        return (double *)calloc(n, sizeof(double));
    }
    return (double *)malloc(n * sizeof(double));
}

/**
 * The allocation of the two-dimensional array used for storing matrices is done
 * in the following way for a matrix in R^(n1*n2):
 * 1. an array of pointers is allocated, one pointer for each row,
 * 2. a 'flat' array of size n1*n2 is allocated to ensure that the memory space
 *   is contigusous,
 * 3. pointers are set for each row to the address of first element.
 */
double **mk_2D_array(size_t n1, size_t n2, bool zero){
    // 1
    double **ret = (double **)malloc(n1 * sizeof(double *));

    // 2
    if (zero) {
        ret[0] = (double *)calloc(n1 * n2, sizeof(double));
    }
    else {
        ret[0] = (double *)malloc(n1 * n2 * sizeof(double));
    }

    // 3
    for (size_t i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}

matrix create_matrix(const size_t rows,const size_t columns){
    auto m = matrix(rows);
    for(auto &r : m){
        r = std::vector<double>(columns);
    }
    return m;
}

matrix transpose(const matrix &m){
    assert(m.size() > 0);
    matrix t = create_matrix(m.at(0).size(), m.size());

    transpose(t, m);
    return t;
}

void transpose(matrix t, const matrix &m){
    for (size_t i = 0; i < m.at(0).size(); i++) {
        for (size_t j = 0; j < m.size(); j++) {
            t.at(i).at(j) = m.at(j).at(i);
        }
    }
}
