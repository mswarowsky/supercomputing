#include <mach.h>

#include <limits>
#include <cassert>
#include <cmath>
#include <iostream>

#define EPSILON  std::numeric_limits<double>::epsilon()


int main() {

    double x = 1./5.0;
    double y = 1./239.0;

    double v_1 = x, v_2 = -(x * x * x)/3.0, v_3 = (x*x*x*x*x)/5.0;
    assert(fabs(v_1 - mach::machElement(1, x)) <= EPSILON);
    assert(fabs(v_2 - mach::machElement(2, x)) <= EPSILON);
    assert(fabs(v_3 - mach::machElement(3, x)) <= EPSILON);

    std::cout << "single element test - PASSED" << std::endl;

    double series = v_1 + v_2 + v_3;
    auto test_series = mach::singleMach(3, x);
    assert(fabs(series - test_series) <= EPSILON);

    std::cout << "series test - PASSED" << std::endl;

    double h_1 = x, h_2 = -(y * y * y)/3.0, h_3 = (y*y*y*y*y)/5.0;
    double series_2 = h_1 + h_2  + h_3;

    double pi = (4 * series - series_2) * 4.0;
    assert(fabs(pi - mach::getPIMachSingleT(3)) <= EPSILON);

    std::cout << "pi single thread test - PASSED" << std::endl;

}
