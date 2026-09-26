// Again, as both cpp files make use of standard_mul and strassen_mul

// To avoid pasting the this file twice during compilation we write:

#pragma once

#include "matrix.hpp"

// Standard Theta(n^3) multiplication, loop order i-k-j
Matrix standard_mul(const Matrix& A, const Matrix& B);

// Strassen's algorithm header

// (AI-assisted) The cut-off is a parameter, not a compile-time constant, so the experiment
// harness can sweep it without recompiling (objective O2 in the report). Blocks of size <= cutoff are
// multiplied with standard_mul. There is deliberately no default value: every call has to
// state which cut-off it uses, so no measurement can silently run with the wrong one.
Matrix strassen_mul(const Matrix& A, const Matrix& B, int cutoff);