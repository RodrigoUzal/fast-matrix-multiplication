
#include <iostream>
#include <vector>
#include <random>

// The repository will work as follows.
// In my main file I will write the code in c++ for both the regular matrix multiplication 
// algorithm as well as the Strassen's algortihm which we will be testing it against and analysing.

// Results will be collected and saved in a cvs file as to facilitate plotting later on with
// python code.

// HERE WE WILL INITIALISE THE CSV FILE

// MATRIX STRUCTURE DEFINITION

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

void fill_random(Matrix& mat){
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0); // All elements are between 0 and 1
    for(int i = 0; i < mat.rows * mat.cols; i++){
        mat.data[i] = dist(rng); // We fill each element of the matrix with a random value
    }
}

void print(Matrix& mat){
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            std::cout << mat(i,j) << " ";
        }
        std::cout << '\n'; // New line separator
    }
    std::cout << std::endl;
}

int main(){
    int n = 10;
    Matrix A(n, n);
    fill_random(A);
    print(A);
}

// MAIN ALGORITHM FUNCTIONS

