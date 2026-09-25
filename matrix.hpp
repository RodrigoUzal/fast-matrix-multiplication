// Here I will store the definition of the structure matrix and the helper functions related to it.
// I defined it separatedly since both cpp files need to make use of it.

#include <iostream>
#include <random>
#include <vector>

#pragma once

struct Matrix{
    int rows;
    int cols;
    std::vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    // Operator () definiton for accessing elements: A(row, col)

    // If matrix has been declared as a type Matrix then this operator is called which returns the &
    inline double& operator()(int r, int c){
        return data[r * cols + c];
    }
    
    // if matrix has been declared as type const Matrix then this operator is called that returns the value itself.
    inline double operator()(int r, int c) const{
        return data[r * cols + c];
    }
};

// We also define a matrix automatic filler which will speed up the experimentation later on

inline void fill_random(Matrix& mat){
    static std::random_device rd; // Hardware entropy source
    static std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0); // All elements are between 0 and 1
    for(int i = 0; i < mat.rows * mat.cols; i++){
        mat.data[i] = dist(rng); // We fill each element of the matrix with a random value
    }
}

inline void print(Matrix& mat){
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            std::cout << mat(i,j) << " ";
        }
        std::cout << '\n'; // New line separator
    }
    std::cout << std::endl;
}


// These are the two helper functions necessary for the implementation of Strassen

inline Matrix add(const Matrix& A, const Matrix& B){
    if(A.rows != B.rows || A.cols != B.cols){
        throw std::invalid_argument("Dimensions mismatch for addition");
    }
    Matrix C(A.rows, A.cols);
    const int n = A.rows * A.cols;
    for (int i = 0; i < n; i++){
        C.data[i] = A.data[i] + B.data[i];
    }
    return C;
}

inline Matrix subtract(const Matrix& A, const Matrix& B){
    if(A.rows != B.rows || A.cols != B.cols){
        throw std::invalid_argument("Dimensions mismatch for subtraction");
    }
    Matrix C(A.rows, A.cols);
    const int n = A.rows * A.cols;
    for (int i = 0; i < n; i++){
        C.data[i] = A.data[i] - B.data[i];
    }
    return C;
}

// It returns the block starting at row0 and col0 of size "size"
inline Matrix submatrix(const Matrix& A, int row0, int col0, int size){
    Matrix S(size, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            S(i, j) = A(row0 + i; col0 + j);
        }
    }
    return S;
}

// Reconstructs matrix C
inline void write_block(Matrix& C, const Matrix& block, int row0, int col0){
    for(int i = 0; i < block.rows; i++){
        for(int j = 0; j < block.cols; j++){
            C(row0 + i, col0 + j) = block(i, j);
        }
    }
}