// Here we store the functions of the main algorithms. Standard multiplicaiton and Strassen's algorithm.
// Both live in the same file as Strassen's will make use of standard multiplication.

#include "algorithms.hpp"

// We must define a constant for the base case of Strassen
constexpr int STRASSEN_CUTOFF = 64;

Matrix standard_mul(const Matrix& A, const Matrix& B){
    // Check multiplication condition is fulfilled.
    if( A.cols != B.rows){
        throw std::invalid_argument("Dimension mismatch: A cols must equal B rows");
    }
    Matrix C(A.rows, B.cols); // Initialise the result matrix C
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

// Define the recursive function itself
static Matrix strassen_recursive(const Matrix& A, const Matrix& B){
    // h now works for splitting both A and B into blocks as they have the same dimensions
    // This will not be the case if we deal with non perfect cases later on.
    int n = A.rows;
    // Base case
    if(n <= STRASSEN_CUTOFF){
        return standard_mul(A, B);
    }
    int h = n/2;
    // We split A and B into blocks
    Matrix A11 = submatrix(A, 0, 0, h);
    Matrix A12 = submatrix(A, 0, h, h);
    Matrix A21 = submatrix(A, h, 0, h);
    Matrix A22 = submatrix(A, h, h, h);
    Matrix B11 = submatrix(B, 0, 0, h);
    Matrix B12 = submatrix(B, 0, h, h);
    Matrix B21 = submatrix(B, h, 0, h);
    Matrix B22 = submatrix(B, h, h, h);
    // Now we compute the M matrices defined by Strassen and call recusion for each multiplication
    Matrix M1 = strassen_recursive(add(A11, A22), add(B11, B22));
    Matrix M2 = strassen_recursive(add(A21, A22), B11);
    Matrix M3 = strassen_recursive(A11, subtract(B12, B22));
    Matrix M4 = strassen_recursive(A22, subtract(B21, B11));
    Matrix M5 = strassen_recursive(add(A11, A12), B22);
    Matrix M6 = strassen_recursive(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassen_recursive(subtract(A12, A22), add(B21, B22));

    // Construct the solution matrix C
    Matrix C11 = subtract(add(add(M1, M4),M7), M5);
    Matrix C12 = add(M3, M5);
    Matrix C21 = add(M2, M4);
    Matrix C22= add(subtract(M1, M2), add(M3, M6));

    Matrix C(n, n);
    write_block(C, C11, 0, 0);
    write_block(C, C12, 0, h);
    write_block(C, C21, h, 0);
    write_block(C, C22, h, h);
    return C;
}

static bool is_power_of_two(int n){
    return n > 0 && (n & (n-1)) == 0;
    // Here n & (n-1) is a bit test, if n is a power of 2 then it has the form 1000 and flipping it
    // render its "inverse" 0111 so the AND leads to 0
}

Matrix strassen_mul(const Matrix& A, const Matrix& B){
    // An important design consideration that I realised while studying the algorithm is that, since we
    // are forced to divide the matrices into equal sized blocks, the easiest case to handle are 
    // square matrices. All testing will firstly be carried out with squared matrices. May be later on
    // I'll implement a disfferent version of strassen for non-square matrices.
    if(A.cols != B.rows){
        throw std:: invalid_argument("Dimension mismatch: A cols must equal B rows");
    }
    if(A.rows != A.cols || B.rows != B.cols){
        throw std::invalid_argument("strassen_mul currently requires squared matrices");
    }
    if(!is_power_of_two(A.rows)){
        throw std::invalid_argument("strasse_mul currently requires a size that is a power of 2");
    }
    return strassen_recursive(A, B);
}
