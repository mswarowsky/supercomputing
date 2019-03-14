
#ifndef PROJECT_1_MACH_H
#define PROJECT_1_MACH_H

#include <cmath>

namespace mach {

    /**
     * Compute on machin element
     * @param i index of the element
     * @param x the input of the series, /in [-1,1]
     * @return the value of element with given index and input
     */
    double machElement(long i, double x);

    /**
     * Compute the machin series over n elements for the given x.
     * @param n
     * @param x
     * @return
     */
    double singleMach(long n, double x);

    /**
     * compute PI single threaded with the machin method
     * @param n number of series elements that should be used
     * @return
     */
    double getPIMachSingleT(long n);

    /**
     * Calculates PI based on the the arctan(1/5) and the arctan(1/239)
     * @param s_1 arctan(1/5)
     * @param s_2 arctan(1/239)
     * @return approximation of PI
     */
    double getPIFromArctans(double s_1, double s_2);
}

#endif //PROJECT_1_MACH_H
