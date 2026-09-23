// Here we store the functions of the main algorithms. Standard multiplicaiton and Strassen's algorithm.
// Both live in the same file as Strassen's will make use of standard multiplication.

#include "algorithms.hpp"

Matrix standard_mul(const Matrix& A, const Matrix& B){
    // Check multiplication condition is fulfilled.
    if( A.cols != B.rows){
        throw std::invalid_argument("Dimension mismatch: A cols must equal B rows");
    }

    // Initialise the result matrix C
    Matrix C(A.rows, B.cols);
    for(int i = 0; i < A.rows; i++){
        for(int k = 0; k < A.cols; k++){
            double temp = A(i, k);
            for(int j = 0; j < B.cols; j++){
                C(i, j) += temp * B(k, j);
            }
        }
    }

    return C;
}

// Here I'll implement Strassen's algorithm

Matrix strassen_mul(const Matrix& A, const Matrix& B){
    // To do
    (void)A;
    (void)B; // For now just ignore both parameters.
    throw std::logic_error("strassen_mul is not implemented yet");
}