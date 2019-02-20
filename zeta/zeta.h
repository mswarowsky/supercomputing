/**
 * All functions fot the zeta approximation
 */

#ifndef PROJECT_1_ZETA_H
#define PROJECT_1_ZETA_H
namespace zeta {
    /**
 * Compute the single element of the zeta series
 * @param i index of the series element
 * @return
 */
    double zetaElement(int i);

/**
 * Get PI from a value of the zeta Series
 * @param s the sum of single zeta series elements
 * @return approximation of Pi
 */
    double getPIfromZetaSeries(double s);


    /**
     * Compute the zeta series single threaded
     * @param n the number of steps
     * @return the value of the series
     */
    double singleZeta(int n);
}




#endif //PROJECT_1_ZETA_H
