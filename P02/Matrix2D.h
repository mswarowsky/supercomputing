#ifndef POISSON_MATRIX2D_H
#define POISSON_MATRIX2D_H

#include <vector>

template <class T>
class Matrix2D {
public:
    Matrix2D(size_t x, size_t y) : data(x*y), columns(x), rows(y) {}
    T &operator()(size_t x, size_t y){
        return data.at(x * columns + y);
    }
    T* row_ptr(size_t i){
        return data.data() + i * columns;
    }
    T* base_ptr() {data.data();}
    size_t size() { return data.size();}
    size_t getColumns() { return  columns;}
    size_t getRows() { return rows;}


private:
    std::vector<T> data;
    size_t columns;
    size_t rows;
};

#endif //POISSON_MATRIX2D_H
