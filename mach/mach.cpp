#include "mach.h"


#include <cassert>
#include <cmath>
#include <iostream>


namespace mach {

    double machElement(long i, double x) {

        //fancy but faster way than pow to alternate the the minus, thanks to unit tests :D
        double sign = 1.0 - (2.0 * ((i & 0x1) ^ 0x1));
        double factor = 2.0 * i - 1.0;

        return sign * pow(x, factor)/factor;
    }

    double singleMach(long n, double x) {
        assert(n > 0);
        assert(x >= -1.0 && x <= 1.0);


        double sum = 0.0;
        for (long i = 1; i <= n; i++){
            sum += machElement(i, x);
        }
        return sum;
    }

    double getPIMachSingleT(long n) {
        double series_1 = singleMach(n, 1./5.);
        double series_2 = singleMach(n, 1./239.);

        double result = (4. * series_1 - series_2) * 4.;

        return result;
    }
}

