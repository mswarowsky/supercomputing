// Unit Test for zeta

#include <zeta.h>
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>

#define EPSILON  std::numeric_limits<double>::epsilon()

int main() {

    double v_1 = 1.0, v_2 = 1./4.0, v_3 = 1./9.0;
    assert(fabs(v_1 - zeta::zetaElement(1)) <= EPSILON);
    assert(fabs(v_2 - zeta::zetaElement(2)) <= EPSILON);
    assert(fabs(v_3 - zeta::zetaElement(3)) <= EPSILON);

    std::cout << "single element test - PASSED" << std::endl;

    auto series = zeta::singleZeta(3);
    double s_3 = v_1 + v_2 + v_3;
    assert(fabs(series - s_3) <= EPSILON);

    std::cout << "single thread series sum test - PASSED" << std::endl;

    auto s_3_pi = sqrt(s_3 * 6.0);
    assert(fabs(zeta::getPIfromZetaSeries(s_3) - s_3_pi) <= EPSILON);

    std::cout << "pi test - PASSED" << std::endl;

    return 0;
}

