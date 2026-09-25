
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream> // file streams
#include <chrono> // clocks

#include "matrix.hpp"
#include "algorithms.hpp"

// The repository will work as follows.
// In my main file I will write the code in c++ for both the regular matrix multiplication 
// algorithm as well as the Strassen's algortihm which we will be testing it against and analysing.

// Results will be collected and saved in a cvs file as to facilitate plotting later on with
// python code.

int main(){
    /*// HERE WE WILL INITIALISE THE CSV FILE (I was helped by AI to build the measurement logic here)
    std::ofstream results("results.csv"); // creates the file or overwrites it if it exists
    results << "algorithm,n,run,seconds\n"; // header row

    // Let us run a test

    int n = 256;
    int run = 1;

    Matrix A(n,n);
    Matrix B(n,n);
    fill_random(A);
    fill_random(B);

    // Time one multiplication (I was helped by AI to build the measurement logic here)
    auto start = std::chrono::steady_clock::now(); // Start timer
    Matrix C = standard_mul(A, B);
    auto end = std::chrono::steady_clock::now(); // End timer
    std::chrono::duration<double> elapsed = end - start; // Compute total time passed

    // Now we write the results into the csv
    results << "standard" << "," << n << "," << run << "," << elapsed.count() << "\n";

    results.close(); */

    int n = 512;
    Matrix A(n,n);
    Matrix B(n,n);
    fill_random(A);
    fill_random(B);
    
    Matrix C_std = standard_mul(A, B);
    Matrix C_str = strassen_mul(A, B);

    double worst = 0.0;
    for(int i = 0; i < n * n; i++){
    worst = std::max(worst, std::fabs(C_std.data[i] - C_str.data[i]));
    }
    std::cout << "max difference = " << worst << "\n";
    return 0;

}
