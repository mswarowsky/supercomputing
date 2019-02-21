
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
}

#endif //PROJECT_1_MACH_H
