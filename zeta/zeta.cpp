#include "zeta.h"
#include <cmath>

namespace zeta
{
    double zetaElement(long i){
        return  1.0/ static_cast<double>(i * i);
    }

    double getPIfromZetaSeries(double s) {
        return sqrt(s * 6.0);
    }

    double singleZeta(long n) {
        double zeta_series = 0.0;

        for(long i = 1; i <= n; i++){
            zeta_series += zeta::zetaElement(i);
        }

        return zeta_series;
    }
}

