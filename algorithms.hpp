// Again, as both cpp files make use of standard_mul and strassen_mul

// To avoid pasting the this file twice during compilation we write:

#pragma once

#include "matrix.hpp"

// Standard Theta(n^3) multiplication, loop order i-k-j
Matrix standard_mul(const Matrix& A, const Matrix& B);

// Strassen's algorithm header will go here.