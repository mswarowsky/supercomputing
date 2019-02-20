#include "zeta.h"
#include <cmath>

namespace zeta
{
    double zetaElement(int i){
        return  1.0/ static_cast<double>(i * i);
    }

    double getPIfromZetaSeries(double s) {
        return sqrt(s * 6.0);
    }

    double singleZeta(int n) {
        double zeta_series = 0.0;

        for(int i = 1; i <= n; i++){
            zeta_series += zeta::zetaElement(i);
        }

        return zeta_series;
    }
}

