#include "Matrix2D.h"


template<class T>
T &Matrix2D<T>::operator()(size_t x, size_t y) {
    return data.at(y * columns + x);
}

template<class T>
T* Matrix2D<T>::row_ptr(size_t i) {
    return data.data() + i * columns;
}


